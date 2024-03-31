import logging
import os

import numpy as np
import optuna
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear

from baseline_evals.feature_selection import variational_selection

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        """
        This function initializes the MyDataset class.

        Args:
            X (np.ndarray): The features (data points).
            y (np.ndarray): The labels (target values).
            transform (torchvision.transforms, optional): A transform to apply to the data. Defaults to None.
        """
        self.X = X
        self.y = y
        self.transform = transform  # Optional transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]

        # Apply transform if provided
        if self.transform:
            X_sample = self.transform(X_sample)  # Convert to PyTorch tensor

        return X_sample, y_sample


class MLP(torch.nn.Module):
    def __init__(self, input_sz, num_classes, proj_dim, hidden_channels, dropout):
        super().__init__()
        torch.manual_seed(12345)
        self.proj = Linear(input_sz, proj_dim)
        self.dropout = dropout

        assert (
            proj_dim == hidden_channels[0]
        ), "Projection dim must match first hidden layer dim"

        self.hidden_layers = torch.nn.ModuleList(
            [
                Linear(hidden_channels[i], hidden_channels[i + 1])
                for i in range(len(hidden_channels) - 1)
            ]
        )

        self.classifier = Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        # apply projection layer
        x = self.proj(x)
        x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # apply all hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


class MLPTrainer(L.LightningModule):  # Fixed the import of LightningModule
    def __init__(self, net, lr, l1_lambda, l2_lambda):  # Changed 'Net' to 'net'
        super().__init__()
        self.net = net  # Changed 'Net' to 'net'
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.metrics = {
            "acc": [],
            "f1_macro": [],
            "f1_weighted": [],
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        loss = F.cross_entropy(y_pred, y)  # Changed 'torch.nn.functional' to 'F'
        # L1 regularization for self.net.proj layer
        l1_reg = torch.tensor(0.0).to(self.device)
        for param in self.net.proj.parameters():
            l1_reg += torch.norm(param, 1)

        loss += self.l1_lambda * l1_reg

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        val_loss = F.cross_entropy(y_pred, y)  # Changed 'torch.nn.functional' to 'F'
        # move to cpu and calculate metrics
        y_pred = y_pred.to(torch.device("cpu"))
        val_f1 = f1_score(
            y.to(torch.device("cpu")), y_pred.argmax(dim=1), average="weighted"
        )  # Fixed the calculation of f1_score
        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        # move to cpu and calculate metrics
        y = y.to(torch.device("cpu"))
        y_pred = y_pred.to(torch.device("cpu"))
        test_acc = accuracy_score(y, y_pred.argmax(dim=1))
        test_f1_macro = f1_score(y, y_pred.argmax(dim=1), average="macro")
        test_f1_weighted = f1_score(y, y_pred.argmax(dim=1), average="weighted")

        self.metrics["acc"].append(test_acc)
        self.metrics["f1_macro"].append(test_f1_macro)
        self.metrics["f1_weighted"].append(test_f1_weighted)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2_lambda
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def mlp_eval(
    X,
    y,
    random_state=3,
    n_evals=5,
    n_trials=100,
    val_test_size=0.4,
    n_features=5000,
    verbose=True,
):
    best_results = {
        "acc": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "acc_std": 0.0,
        "f1_macro_std": 0.0,
        "f1_weighted_std": 0.0,
    }

    def objective(trial):
        nonlocal best_results

        print(f"Trial {trial.number} / {n_trials}")

        sss = StratifiedShuffleSplit(
            n_splits=n_evals, test_size=val_test_size, random_state=random_state
        )
        num_layers = 1  # trial.suggest_int("num_layers", 1, 3)
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "l1_lambda": trial.suggest_float("l1_lambda", 1e-4, 1, log=True),
            "l2_lambda": 5e-4,  # trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True),
            "batch_sz": 64,  # trial.suggest_categorical("batch_sz", [32, 64, 128]),
            "proj_dim": 54,  # trial.suggest_int("proj_dim", 32, 128),
            "dropout": trial.suggest_float("dropout", 0.05, 0.8),
            "num_layers": num_layers,
            "hidden_channels": [
                46
            ],  # [trial.suggest_int("hidden_channels", 32, 128) for _ in range(num_layers)],
        }

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_weighted = np.zeros(n_evals)

        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            print(f"Eval {i+1} / {n_evals}")
            # split test_idx into val_idx and test_idx
            val_idx = test_index[: len(test_index) // 2]
            test_idx = test_index[len(test_index) // 2 :]

            X_train = X[train_index]
            X_val = X[val_idx]
            X_test = X[test_idx]

            # feature pre-selection
            if n_features:
                select_mask, select_idx = variational_selection(
                    X_train, y[train_index], n_features
                )
                X_train = X_train[:, select_mask]
                X_val = X_val[:, select_mask]
                X_test = X_test[:, select_mask]

            # scale features
            std_scale = StandardScaler()
            X_train = std_scale.fit_transform(X_train)
            X_val = std_scale.transform(X_val)
            X_test = std_scale.transform(X_test)

            train_loader = torch.utils.data.DataLoader(
                MLPDataset(torch.Tensor(X_train), torch.tensor(y[train_index])),
                batch_size=params["batch_sz"],
                num_workers=os.cpu_count() - 1,
            )
            val_loader = torch.utils.data.DataLoader(
                MLPDataset(torch.Tensor(X_val), torch.tensor(y[val_idx])),
                batch_size=params["batch_sz"],
                num_workers=os.cpu_count() - 1,
            )
            test_loader = torch.utils.data.DataLoader(
                MLPDataset(torch.Tensor(X_test), torch.tensor(y[test_idx])),
                batch_size=params["batch_sz"],
                num_workers=os.cpu_count() - 1,
            )

            # define model
            mlp = MLP(
                input_sz=X_train.shape[1],
                num_classes=len(np.unique(y)),
                proj_dim=params["proj_dim"],
                hidden_channels=[params["proj_dim"]] + params["hidden_channels"],
                dropout=params["dropout"],
            )
            mlp_lightning_module = MLPTrainer(
                net=mlp,
                lr=params["lr"],
                l1_lambda=params["l1_lambda"],
                l2_lambda=params["l2_lambda"],
            )

            trainer = L.Trainer(
                max_epochs=50,
                callbacks=[L.callbacks.EarlyStopping(monitor="val_loss", mode="min")],
                log_every_n_steps=-1,
                enable_progress_bar=False,
            )

            # train and test the model
            trainer.fit(
                model=mlp_lightning_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            trainer.test(model=mlp_lightning_module, dataloaders=test_loader)

            accs[i] = torch.tensor(mlp_lightning_module.metrics["acc"]).mean()
            f1_macros[i] = torch.tensor(mlp_lightning_module.metrics["f1_macro"]).mean()
            f1_weighted[i] = torch.tensor(
                mlp_lightning_module.metrics["f1_weighted"]
            ).mean()

            # if after 2 evals this doesnt seem promising, break
            if i >= 2 and (
                f1_weighted[:i].mean()
                < (best_results["f1_weighted"] - 2 * best_results["f1_weighted_std"])
            ):
                print(
                    f"Pruning trial after {i} evals, cause {f1_weighted[:i].mean()} < {best_results['f1_weighted']}"
                )
                break

        acc = np.mean(accs)
        f1_macro = np.mean(f1_macros)
        f1_weighted = np.mean(f1_weighted)
        acc_std = np.std(accs)
        f1_macro_std = np.std(f1_macros)
        f1_weighted_std = np.std(f1_weighted)

        current_result = acc + f1_macro + f1_weighted

        if (
            current_result
            > best_results["acc"]
            + best_results["f1_weighted"]
            + best_results["f1_macro"]
        ):
            best_results["acc"] = acc
            best_results["f1_macro"] = f1_macro
            best_results["f1_weighted"] = f1_weighted
            best_results["acc_std"] = acc_std
            best_results["f1_macro_std"] = f1_macro_std
            best_results["f1_weighted_std"] = f1_weighted_std

        return current_result

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print best parameters and results
        print(f"{study.best_value=}, {study.best_params=}")
        print(
            f"| MLP | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )

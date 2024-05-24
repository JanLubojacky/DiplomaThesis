import logging
import os

import numpy as np
import optuna
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import Linear

from baseline_evals.feature_selection import class_variational_selection

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
            X_sample = self.transform(X_sample)

        return X_sample, y_sample


class MLP(torch.nn.Module):
    def __init__(self, input_sz, num_classes, proj_dim, hidden_channels, dropout):
        super().__init__()
        torch.manual_seed(12345)

        self.proj = Linear(input_sz, proj_dim)

        self.dropout = dropout
        self.hidden_layer = Linear(
            proj_dim,
            hidden_channels,
            weight_initializer="kaiming_uniform",
        )

        self.classifier = Linear(hidden_channels, num_classes, "kaiming_uniform")

    def forward(self, x):
        # apply projection layer
        x = self.proj(x)
        x = F.elu(x)

        # apply hidden layer
        x = self.hidden_layer(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # apply classifier
        return self.classifier(x)

    def projection_layer_l1_norm(self):
        """
        Returns the l1 norm of the projection layer
        """
        return torch.norm(self.proj.weight, 1)

    def projection_layer_inner_mat_reg(self):
        """
        Return the sum of inner products of each of the projection layers
        where inner product is ||W * W^T||_1 + ||W||_2^2
        """
        return (
            torch.norm(torch.matmul(self.proj.weight, self.proj.weight.T), 1)
            + torch.norm(self.proj.weight, 2) ** 2
        )


class MLPTrainer(L.LightningModule):
    def __init__(self, net, lr, l2_lambda, reg_lambda=0.001, regularization=None):
        super().__init__()
        self.net = net
        self.lr = lr
        self.l2_lambda = l2_lambda
        self.reg_lambda = reg_lambda
        self.regularization = regularization
        self.metrics = {
            "acc": [],
            "f1_macro": [],
            "f1_weighted": [],
        }

    def training_step(self, batch, batch_idx):
        # ce loss
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        loss = F.cross_entropy(y_pred, y)

        if self.regularization == "l1":
            # print("l1 regularization")
            l1_reg = self.net.projection_layer_l1_norm()
            loss += self.reg_lambda * l1_reg
        elif self.regularization == "inner_mat":
            # print("inner mat regularization")
            inner_mat_reg = self.net.projection_layer_inner_mat_reg()
            loss += self.reg_lambda * inner_mat_reg

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
    # val_test_size=0.4,
    # n_features_preselect=10000,
    n_features=5000,
    verbose=True,
    reg_type="l1",
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
        nonlocal best_results  # , select_masks

        if verbose:
            print(f"Trial {trial.number} / {n_trials}")

        skf = StratifiedKFold(n_splits=n_evals)

        # params = {
        #     "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        #     "l1_lambda": trial.suggest_float("l1_lambda", 1e-4, 5e-2, log=True),
        #     "l2_lambda": trial.suggest_float("l2_lambda", 1e-5, 1e-3, log=True),
        #     "batch_sz": 128,  # trial.suggest_categorical("batch_sz", [32, 64, 128]),
        #     "proj_dim": 64,  # trial.suggest_int("proj_dim", 32, 256),
        #     "dropout": 0.5,  # trial.suggest_float("dropout", 0.0, 0.7),
        #     "hidden_channels": 150,  # trial.suggest_int("hidden_channels", 32, 256),
        #     "regularization": reg_type,  # trial.suggest_categorical("regularization", ["l1", "inner_mat"]),
        # }

        # n_features = trial.suggest_int("n_features", 100, 1800)

        params = {
            "lr": 1e-3,  # trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "l2_lambda": 5e-4,  # trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True),
            "l1_lambda": 0.001,
            "batch_sz": 128,
            "proj_dim": 64,
            "dropout": 0.5,
            "hidden_channels": 86,
            "regularization": reg_type,  # trial.suggest_categorical("regularization", ["l1", "inner_mat"]),
        }

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_weighteds = np.zeros(n_evals)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {i + 1} / {n_evals}")

            val_idx, test_idx = train_test_split(
                test_index,
                test_size=0.5,
                random_state=random_state,
                stratify=y[test_index],
            )

            X_train = X[train_index]
            X_val = X[val_idx]
            X_test = X[test_idx]

            if n_features:
                # feature selection
                select_idx = class_variational_selection(
                    X_train, y[train_index], n_features
                )
                # select_idx = variance_filtering(X_train, n_features)
                X_train = X_train[:, select_idx]
                X_val = X_val[:, select_idx]
                X_test = X_test[:, select_idx]

            # scale features
            std_scale = StandardScaler()
            X_train = std_scale.fit_transform(X_train)
            X_val = std_scale.transform(X_val)
            X_test = std_scale.transform(X_test)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

            train_loader = torch.utils.data.DataLoader(
                MLPDataset(X_train, y[train_index]),
                batch_size=params["batch_sz"],
                num_workers=os.cpu_count() - 1,
            )
            val_loader = torch.utils.data.DataLoader(
                MLPDataset(X_val, y[val_idx]),
                batch_size=params["batch_sz"],
                num_workers=os.cpu_count() - 1,
            )
            test_loader = torch.utils.data.DataLoader(
                MLPDataset(X_test, y[test_idx]),
                batch_size=params["batch_sz"],
                num_workers=os.cpu_count() - 1,
            )

            # define model
            mlp = MLP(
                input_sz=X_train.shape[1],
                num_classes=len(np.unique(y)),
                proj_dim=params["proj_dim"],
                hidden_channels=params["hidden_channels"],
                dropout=params["dropout"],
            )
            mlp_lightning_module = MLPTrainer(
                net=mlp,
                lr=params["lr"],
                reg_lambda=params["l1_lambda"],
                l2_lambda=params["l2_lambda"],
                regularization=params["regularization"],
            )

            trainer = L.Trainer(
                max_epochs=50,
                callbacks=[L.callbacks.EarlyStopping(monitor="val_loss", mode="min")],
                log_every_n_steps=5,
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
            f1_weighteds[i] = torch.tensor(
                mlp_lightning_module.metrics["f1_weighted"]
            ).mean()

            # if after 2 evals this doesnt seem promising, break
            if i >= 1 and (
                f1_weighteds[:i].mean()
                < (best_results["f1_weighted"] - 2 * best_results["f1_weighted_std"])
            ):
                if verbose:
                    print(
                        f"Pruning trial after {i + 1} evals, cause {f1_weighteds[:i].mean()} < {best_results['f1_weighted']}"
                    )
                break

        if verbose:
            print(accs)
            print(f1_macros)
            print(f1_weighteds)

        acc = np.mean(accs)
        f1_macro = np.mean(f1_macros)
        f1_weighted = np.mean(f1_weighteds)
        acc_std = np.std(accs)
        f1_macro_std = np.std(f1_macros)
        f1_weighted_std = np.std(f1_weighteds)

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

            if verbose:
                print("New best results")
                print(best_results)

            # save the best model
            # torch.save(mlp.state_dict(), "mlp_best_model.pth")

        return current_result

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print best parameters and results
        print(f"{study.best_value=}, {study.best_params=}")
        print(
            f"| MLP regularization : {reg_type} | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )

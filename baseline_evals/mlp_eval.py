import numpy as np
import optuna
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn import Linear


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

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        loss = F.cross_entropy(y_pred, y)  # Changed 'torch.nn.functional' to 'F'
        # L1 regularization for self.net.proj layer
        l1_reg = torch.tensor(0.0)
        for param in self.net.proj.parameters():
            l1_reg += torch.norm(param, 1)
            loss += self.l1_lambda * l1_reg

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        val_loss = F.cross_entropy(y_pred, y)  # Changed 'torch.nn.functional' to 'F'
        val_f1 = f1_score(
            y, y_pred.argmax(dim=1), average="weighted"
        )  # Fixed the calculation of f1_score
        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        test_acc = accuracy_score(y, y_pred.argmax(dim=1))
        test_f1_macro = f1_score(y, y_pred.argmax(dim=1), average="macro")
        test_f1_weighted = f1_score(y, y_pred.argmax(dim=1), average="weighted")
        return test_acc, test_f1_macro, test_f1_weighted
        self.log("test_f1", test_f1_weighted)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2_lambda
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler_config": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def mlp_eval(
    X, y, random_state=3, n_evals=5, n_trials=100, test_size=0.2, verbose=True
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

        print(f"{trial.number} / {n_trials}")

        sss = StratifiedShuffleSplit(
            n_splits=n_evals, test_size=test_size, random_state=random_state
        )
        proj_dim = trial.suggest_int("proj_dim", 32, 256)
        hidden_channels = [
            trial.suggest_int("hidden_channels", 32, 256)
            for _ in range(trial.suggest_int("num_layers", 1, 3))
        ]
        # define model
        net = MLP(
            input_sz=X.shape[1],
            num_classes=len(np.unique(y)),
            proj_dim=proj_dim,
            hidden_channels=[proj_dim] + hidden_channels,
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
        )
        trainer = MLPTrainer(
            net,
            lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            l1_lambda=trial.suggest_float("l1_lambda", 1e-2, 10, log=True),
            l2_lambda=trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True),
        )

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_weighted = np.zeros(n_evals)

        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            dataset_train = MLPDataset(torch.tensor(X_train), torch.tensor(y_train))
            dataset_test = MLPDataset(torch.tensor(X_test), torch.tensor(y_test))
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
            trainer.fit(train_loader, test_loader)

            acc, f1_m, f1_w = trainer.test(test_loader)
            accs[i] = acc
            f1_macros[i] = f1_m
            f1_weighted[i] = f1_w

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

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        print(
            f"| MLP | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )

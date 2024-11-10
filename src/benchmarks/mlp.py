import optuna

from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager
from src.models.mlp import MLPModel, MLPLightningModule, MLPDataset

import torch
import pytorch_lightning as L


class MLPEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
        params: dict = {},
    ):
        super().__init__(data_manager, n_trials, verbose)
        self.params = params
        self.model = None
        self.lightning_model = None

    def create_model(self, trial: optuna.Trial):
        """Create and save a model instance given parameters in the current trial"""
        input_size = self.data_manager.get_feature_dim()
        num_classes = self.data_manager.get_num_classes()

        # Define hyperparameters
        if self.params["lr_range"] is None:
            lr = self.params["lr"]
        else:
            lr = trial.suggest_float(
                "lr", self.params["lr_range"][0], self.params["lr_range"][1], log=True
            )
        if self.params["l2_lambda_range"] is None:
            self.params["l2_lambda"] = 5e-4
        else:
            l2_lambda = trial.suggest_float(
                "l2_lambda",
                self.params["l2_lambda_range"][0],
                self.params["l2_lambda_range"][1],
                log=True,
            )
        if self.params["dropout_range"] is None:
            dropout = self.params["dropout"]
        else:
            dropout = trial.suggest_float(
                "dropout", self.params["dropout_range"][0], self.params["dropout_range"][1], 0.5
            )
        if self.params["hidden_channels_range"] is None:
            hidden_channels = self.params["hidden_channels"]
        else:
            hidden_channels = trial.suggest_int(
                "hidden_channels",
                self.params["hidden_channels_range"][0],
                self.params["hidden_channels_range"][1],
            )

        # Create model instance
        self.model = MLPModel(
            input_sz=input_size,
            num_classes=num_classes,
            proj_dim=self.params["proj_dim"],
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        # Create lightning module
        self.lightning_model = MLPLightningModule(net=self.model, lr=lr, l2_lambda=l2_lambda)

    def train_model(self, train_x, train_y) -> None:
        """Train model implementation"""
        # Create data loader
        train_dataset = MLPDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )

        # Train model
        trainer = L.Trainer(
            max_epochs=self.params["max_epochs"],
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(self.lightning_model, train_loader)

    def test_model(self, test_x, test_y) -> dict:
        """Test model implementation"""

        # Create test dataset and loader
        test_dataset = MLPDataset(test_x, test_y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.params["batch_size"])

        # Get predictions
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.view(x.size(0), -1)
                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())

        # Calculate and return metrics
        return self._calculate_metrics(test_y, all_preds)

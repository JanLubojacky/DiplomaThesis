import optuna
import torch

from src.models.birgat import BiRGAT
from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager
from src.gnn_utils.gnn_trainer import GNNTrainer


class BiRGATEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
        params: dict = {},
    ):
        """ """
        super().__init__(data_manager, n_trials, verbose)
        self.params = params
        self.model = None

        data, _, _, _ = data_manager.get_split(0)
        self.n_classes = data.y.unique().shape[0]
        self.in_channels = [data.x_dict[omics].shape[1] for omics in data.x_dict.keys()]
        self.omic_names = list(data.x_dict.keys())


    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""

        # here we can select hyperparameters from the trial later
        params = {
            "hidden_channels": [200, 64, 64, 32, 32],
            "heads": 2,
            "dropout": 0.2,
            "attention_dropout": 0.2,
            "integrator_type": "attention",
            "three_layers": True,
        }

        self.model = BiRGAT(
            omic_channels=self.data_manager.omics,
            feature_names=self.data_manager.feature_names,
            relations=self.data_manager.relations,
            input_dims=self.data_manager.input_dims,
            num_classes=self.data_manager.n_classes,
            hidden_channels=params["hidden_channels"],
            heads=params["heads"],
            dropout=params["dropout"],
            attention_dropout=params["attention_dropout"],
            integrator_type=params["integrator_type"],
            three_layers=params["three_layers"],
        )
        self.trainer = GNNTrainer(
            model=self.model,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3), # this could be later set by the trial
            loss_fn=torch.nn.CrossEntropyLoss(),
            # params={
            #     "l1_lambda": 0.01, # this could be later set by the trial
            # }
        )

    def train_model(self, data, _) -> None:
        """Train model implementation"""
        self.trainer.train(data, self.params["epochs"], self.params["log_interval"])

    def test_model(self, data, _) -> dict:
        """Test model implementation"""
        # y_pred = self.trainer.test(data, data.test_mask).numpy()
        y_pred = self.trainer.best_pred
        y_true = data.y[data.test_mask].numpy()
        return self._calculate_metrics(y_true, y_pred)


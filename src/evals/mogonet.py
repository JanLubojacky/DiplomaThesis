import optuna
import torch

from src.models.mogonet import MOGONET
from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager
from src.gnn_utils.gnn_trainer import GNNTrainer


class MOGONETEvaluator(ModelEvaluator):
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

        self.model = MOGONET(
            omics=self.omic_names,
            in_channels=self.in_channels,
            num_classes=self.n_classes,
            hidden_channels=self.params["encoder_hidden_channels"],
            encoder_type=self.params["encoder_type"],
            dropout=self.params["dropout"],
            integrator_type=self.params["integrator_type"],
            integration_in_dim=self.params["integration_in_dim"],
            vcdn_hidden_channels=self.params["vcdn_hidden_channels"],
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

    def get_feature_importances(self):
        ...

        # per fold:
        #   run permutation importance
        #   measure the drop in loss for each feature
        #   accumulate across all folds


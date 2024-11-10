import optuna

from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager


class MLPEvaluator(ModelEvaluator):
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

    def create_model(self, trial: optuna.Trial):
        """Create and save a model instance given parameters in the current trial"""

    def train_model(self, train_x, train_y) -> None:
        """Train model implementation"""

    def test_model(self, test_x, test_y) -> dict:
        """Test model implementation"""

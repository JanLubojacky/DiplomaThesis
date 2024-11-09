from src.base_classes.omic_data_manager import OmicDataManager
from src.base_classes.evaluator import ModelEvaluator

import optuna

class XGBoostEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
    ):
        super().__init__(data_manager, n_trials, verbose)

    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""
        pass

    def train_model(self, model, train_x, train_y) -> None:
        """Train model implementation"""
        pass

    def test_model(self, model, test_x, test_y) -> dict:
        """Test model implementation"""
        pass

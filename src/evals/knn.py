import optuna
from sklearn.neighbors import KNeighborsClassifier

from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager


class KNNEvaluator(ModelEvaluator):
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
        """Create and return model instance with trial parameters"""
        knn = KNeighborsClassifier(
            n_neighbors=trial.suggest_int(
                "n_neighbors", self.params["k_lb"], self.params["k_ub"]
            ),
            # weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        )
        self.model = knn

    def train_model(self, train_x, train_y) -> None:
        """Train model implementation"""
        self.model.fit(train_x, train_y)

    def test_model(self, test_x, test_y) -> dict:
        """Test model implementation"""
        y_pred = self.model.predict(test_x)
        return self._calculate_metrics(test_y, y_pred)

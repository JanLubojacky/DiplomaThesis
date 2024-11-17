import optuna
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC

from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager


class SVMEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
        mode="linear",
        rfe_step=0.1,
        rfe_n_features=200,
        params={},
    ):
        super().__init__(data_manager, n_trials, verbose)
        self.mode = mode
        self.rfe_step = rfe_step
        self.rfe_n_features = rfe_n_features
        self.params = params
        self.model = None

    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""
        if self.mode == "linear":
            params = {
                "C": trial.suggest_float(
                    "C", self.params["C_lb"], self.params["C_ub"], log=True
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                ),
                "dual": "auto",
            }

            rfe_step_range = self.params.get("rfe_step_range")
            rfe_n_features_range = self.params.get("rfe_n_features_range")
            if rfe_step_range is not None:
                self.rfe_step = trial.suggest_float(
                    "rfe_step", rfe_step_range[0], rfe_step_range[1]
                )
            if rfe_n_features_range is not None:
                self.rfe_n_features = trial.suggest_int(
                    "rfe_n_features", rfe_n_features_range[0], rfe_n_features_range[1]
                )

            rfe = RFE(
                LinearSVC(**params),
                step=self.rfe_step,
                n_features_to_select=self.rfe_n_features,
            )
            if self.params.get("no_rfe"):
                rfe = LinearSVC(**params)

        elif self.mode == "rbf":
            params = {
                "C": trial.suggest_float(
                    "C", self.params["C_lb"], self.params["C_ub"], log=True
                ),
                # "gamma": trial.suggest_float("gamma", self.params["gamma_lb"], self.params["gamma_ub"], log=True),
                "gamma": "scale",
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                ),
                "kernel": "rbf",
            }
            rfe = SVC(**params)

        self.model = rfe

    def train_model(self, train_x, train_y) -> None:
        """Train model implementation"""
        self.model.fit(train_x, train_y)

    def test_model(self, test_x, test_y) -> dict:
        """Test model implementation"""
        y_pred = self.model.predict(test_x)
        return self._calculate_metrics(test_y, y_pred)

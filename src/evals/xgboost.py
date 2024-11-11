import optuna
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager


class XGBoostEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
    ):
        """Initialize XGBoost evaluator"""
        super().__init__(data_manager, n_trials, verbose)
        self.model = None
        self.scaler = StandardScaler()
        self.n_classes = self.data_manager.n_classes

    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""
        params = {
            "verbosity": 1,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "num_class": self.n_classes,
        }

        # Add specific parameters based on booster type
        if params["booster"] in ["gbtree", "dart"]:
            params.update(
                {
                    "max_depth": trial.suggest_int("max_depth", 1, 9),
                    "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                    "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                    "grow_policy": trial.suggest_categorical(
                        "grow_policy", ["depthwise", "lossguide"]
                    ),
                }
            )

        if params["booster"] == "dart":
            params.update(
                {
                    "sample_type": trial.suggest_categorical(
                        "sample_type", ["uniform", "weighted"]
                    ),
                    "normalize_type": trial.suggest_categorical(
                        "normalize_type", ["tree", "forest"]
                    ),
                    "rate_drop": trial.suggest_float("rate_drop", 1e-8, 1.0, log=True),
                    "skip_drop": trial.suggest_float("skip_drop", 1e-8, 1.0, log=True),
                }
            )

        self.params = params

    def train_model(self, train_x, train_y) -> None:
        """Train XGBoost model implementation"""
        # Scale features
        train_x = self.scaler.fit_transform(train_x)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(train_x, label=train_y)

        # Train model
        self.model = xgb.train(self.params, dtrain=dtrain, verbose_eval=False)

    def test_model(self, test_x, test_y) -> dict:
        """Test XGBoost model implementation"""
        # Scale features using fitted scaler
        test_x = self.scaler.transform(test_x)

        # Create DMatrix for prediction
        dtest = xgb.DMatrix(test_x)

        # Get predictions
        y_pred = self.model.predict(dtest)

        # Calculate and return metrics
        return self._calculate_metrics(test_y, y_pred)

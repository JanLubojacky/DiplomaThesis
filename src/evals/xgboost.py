import optuna

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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
        self.n_classes = self.data_manager.n_classes
        self.scaler = StandardScaler()

    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""
        params = {
            "verbosity": 1,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            # "booster": "gblinear",
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

        # Scale data
        train_x = self.scaler.fit_transform(train_x)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(train_x, label=train_y)

        # Train model
        self.model = xgb.train(self.params, dtrain=dtrain, verbose_eval=False)

    def test_model(self, test_x, test_y) -> dict:
        """Test XGBoost model implementation"""

        # Scale data
        test_x = self.scaler.transform(test_x)

        # Create DMatrix for prediction
        dtest = xgb.DMatrix(test_x)

        # Get predictions
        y_pred = self.model.predict(dtest)

        # Calculate and return metrics
        return self._calculate_metrics(test_y, y_pred)

    def get_feature_importances(self, classes: list, parameters=None) -> dict:
        """
        Calculate feature importances across all cross-validation folds.

        Args:
            classes: List of class names for the target variable
            parameters: Optional dictionary of model parameters. If None, uses best_params

        Returns:
            Dictionary mapping feature names to their aggregated importance scores
        """
        # Use provided parameters or fall back to best parameters
        if parameters is None:
            if not hasattr(self, "best_params"):
                raise ValueError(
                    "No parameters provided and no best_params found. Run evaluate() first or provide parameters."
                )
            parameters = self.best_params

        importance_accumulator = {}

        # Prepare model parameters
        model_params = {
            "verbosity": 1,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "booster": "gblinear",  # Since we're using linear booster
            "num_class": self.n_classes,
            **parameters,  # Update with provided/best parameters
        }

        # Iterate through folds
        for fold_idx in range(self.data_manager.n_splits):
            # Get train and test splits for this fold
            train_x, test_x, train_y, test_y = self.data_manager.get_split(fold_idx)

            # get feature names for the current fold
            feature_names = self.data_manager.feature_names

            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

            # Create and train model for this fold
            dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=feature_names)
            model = xgb.train(model_params, dtrain=dtrain, verbose_eval=False)

            # Get predictions for this fold
            dtest = xgb.DMatrix(test_x, feature_names=feature_names)
            y_pred = model.predict(dtest)

            # Calculate and return metrics for this fold
            metrics = self._calculate_metrics(test_y, y_pred)
            print(metrics)

            # For linear booster, get weights for each class
            weights = model.get_score(importance_type="weight")
            print(weights)

            # Aggregate absolute weights across all classes for each feature
            for feature_name in feature_names:
                feature_importance = 0.0
                for class_idx in range(self.n_classes):
                    feature_importance += abs(weights[feature_name][class_idx])

                if feature_name not in importance_accumulator:
                    importance_accumulator[feature_name] = 0
                importance_accumulator[feature_name] += feature_importance

        return importance_accumulator

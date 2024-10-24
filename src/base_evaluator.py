import logging
import os
import polars as pl
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import optuna

from src.data_preprocessing import MultiOmicDataManager, CVSplitManager


class ModelEvaluator(ABC):
    def __init__(
        self,
        data_manager: MultiOmicDataManager,
        cv_manager: CVSplitManager,
        n_trials: int = 30,
        verbose: bool = True,
    ):
        self.data_manager = data_manager
        self.cv_manager = cv_manager
        self.n_trials = n_trials
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.best_results = {
            "acc": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "acc_std": 0.0,
            "f1_macro_std": 0.0,
            "f1_weighted_std": 0.0,
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the evaluator"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def evaluate(self, param_space: dict) -> dict:
        """Main evaluation loop using pre-created splits"""
        self.logger.info("Starting model evaluation")

        # Load data here, load the splits
        # Having a manager that can yield splits is a good idea
        data = self.data_manager.load_data()
        splits = self.cv_manager.load_splits()

        # Running optuna study

        def objective(trial):
            fold_scores = []
            for fold_idx, split in enumerate(splits):
                self.logger.info(f"Evaluating fold {fold_idx + 1}/{len(splits)}")
                model = self.get_model(trial)

                # Train and test model, pass X and y
                self.train_model(model, data, split["train_idx"])
                metrics = self.test_model(model, data, split["test_idx"])
                fold_scores.append(metrics)

            # Aggregate scores across folds
            mean_scores = {
                metric: np.mean([score[metric] for score in fold_scores])
                for metric in fold_scores[0].keys()
            }
            std_scores = {
                f"{metric}_std": np.std([score[metric] for score in fold_scores])
                for metric in fold_scores[0].keys()
            }

            # Think about what metric we can use here? Though using F1 is probably fine
            current_score = mean_scores["f1_weighted"]  # Primary metric

            # Update best results if current score is better
            if current_score > self.best_results["f1_weighted"]:
                self.best_results.update(mean_scores)
                self.best_results.update(std_scores)
                if self.verbose:
                    self.print_best_results()
            self.print_best_results()


            return current_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        return self.best_results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate standard metrics for classification"""
        return {
            "acc": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }

    def print_best_results(self) -> None:
        """Print evaluation results"""
        self.logger.info("Best model performance:")
        self.logger.info(
            f"Accuracy: {self.best_results['acc']:.3f} ± {self.best_results['acc_std']:.3f}"
        )
        self.logger.info(
            f"F1 Macro: {self.best_results['f1_macro']:.3f} ± {self.best_results['f1_macro_std']:.3f}"
        )
        self.logger.info(
            f"F1 Weighted: {self.best_results['f1_weighted']:.3f} ± {self.best_results['f1_weighted_std']:.3f}"
        )

    def save_results(self, results_file: str, row_name: str) -> None:
        """
        Save evaluation results to a CSV file.
        Creates the file if it doesn't exist, otherwise appends to it.

        Args:
            results_file: Path to the CSV file
            row_name: Name identifier for this evaluation row
        """
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        # Prepare the results row
        results_dict = self.best_results.copy()
        results_dict["model"] = row_name
        new_row = pl.DataFrame([results_dict])

        if not os.path.exists(results_file):
            # Create new file
            new_row.write_csv(results_file)
            self.logger.info(f"Results saved to {results_file}")
            return

        df = pl.read_csv(results_file)
        if row_name in df["model"].to_list():  # Update existing row
            df = df.filter(pl.col("model") != row_name)
            df = pl.concat([df, new_row])
        else:  # Append new row
            df = pl.concat([df, new_row])
        new_row.write_csv(results_file)
        self.logger.info(f"Results saved to {results_file}")

    @abstractmethod
    def get_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""
        pass

    @abstractmethod
    def train_model(
        self, model, data: dict[str, np.ndarray], train_idx: np.ndarray
    ) -> None:
        """Train model implementation"""
        pass

    @abstractmethod
    def test_model(
        self, model, data: dict[str, np.ndarray], test_idx: np.ndarray
    ) -> dict:
        """Test model implementation"""
        pass

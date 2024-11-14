import os
from abc import ABC, abstractmethod

import numpy as np
import optuna
import polars as pl
from sklearn.metrics import accuracy_score, f1_score

from src.base_classes.omic_data_loader import OmicDataManager


def is_strictly_better(metrics1: dict, metrics2: dict) -> bool:
    """
    Returns:
        True if metrics2 is strictly better than metrics1
        False otherwise
    """

    # Split metrics and their standard deviations
    base_metrics = ["acc", "f1_macro", "f1_weighted"]

    # Check if any performance metric is worse (lower) in metrics2
    for metric in base_metrics:
        if metrics2[metric] < metrics1[metric]:
            return False

    # Check if at least one metric is strictly better
    is_any_better = any(metrics2[metric] > metrics1[metric] for metric in base_metrics)

    return is_any_better


class ModelEvaluator(ABC):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
    ):
        self.n_trials = n_trials
        self.verbose = verbose
        self.data_manager = data_manager
        self.best_score = 0.0
        self.best_results = {
            "acc": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "acc_std": 0.0,
            "f1_macro_std": 0.0,
            "f1_weighted_std": 0.0,
        }

    def evaluate(self) -> dict:
        """Main evaluation loop using pre-created splits"""

        # Running optuna study
        def objective(trial):
            fold_scores = []
            for fold_idx in range(self.data_manager.n_splits):
                # Get train and test splits
                train_x, test_x, train_y, test_y = self.data_manager.get_split(fold_idx)

                # Creates model and saves it as a class attribute
                self.create_model(trial)

                # Train and test model, pass X and y
                self.train_model(train_x, train_y)
                metrics = self.test_model(test_x, test_y)
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

            current_score = (
                mean_scores["acc"]
                * mean_scores["f1_macro"]
                * mean_scores["f1_weighted"]
            )

            if current_score > self.best_score:
                self.best_score = current_score
                self.best_results.update(mean_scores)
                self.best_results.update(std_scores)
                if self.verbose:
                    print(f"New best score: {current_score:.3f}")
                    self.print_best_results()

            # # Update best results if current score is strictly better
            # if is_strictly_better(self.best_results, mean_scores):
            #     self.best_results.update(mean_scores)
            #     self.best_results.update(std_scores)
            #     if self.verbose:
            #         print(f"New best score: {current_score:.3f}")
            #         self.print_best_results()

            return current_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

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
        print("Best model performance:")
        print(
            f"Accuracy: {self.best_results['acc']:.3f} ± {self.best_results['acc_std']:.3f}"
        )
        print(
            f"F1 Macro: {self.best_results['f1_macro']:.3f} ± {self.best_results['f1_macro_std']:.3f}"
        )
        print(
            f"F1 Weighted: {self.best_results['f1_weighted']:.3f} ± {self.best_results['f1_weighted_std']:.3f}"
        )

    def print_best_parameters(self) -> None:
        print("Best hyperparameters:")
        print(self.best_params)

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

        pretty_names = {
            "acc": "Accuracy",
            "f1_macro": "Macro F1",
            "f1_weighted": "Weighted F1",
        }

        # rename keys to pretty names
        new_row = new_row.rename(pretty_names)

        if not os.path.exists(results_file):
            # Create new file
            new_row.write_csv(results_file)
            print(f"Results saved to {results_file}")
            return

        df = pl.read_csv(results_file)
        if row_name in df["model"].to_list():  # Update existing row
            df = df.filter(pl.col("model") != row_name)
            df = pl.concat([df, new_row])
        else:  # Append new row
            df = pl.concat([df, new_row])
        df.write_csv(results_file)

    @abstractmethod
    def create_model(self, trial: optuna.Trial):
        """Create and safe model instance with trial parameters"""
        pass

    @abstractmethod
    def train_model(self, train_x, train_y) -> None:
        """Train model implementation"""
        pass

    @abstractmethod
    def test_model(self, test_x, test_y) -> dict:
        """Test model implementation"""
        pass

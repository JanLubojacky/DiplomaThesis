import warnings

import numpy as np
import optuna
import optuna.logging
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from baseline_evals.feature_selection import variational_selection

# Before creating the study:
# optuna.logging.set_verbosity(optuna.logging.ERROR)


def svm_lin_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 10,
    n_trials: int = 40,
    C_range: tuple = (1e-3, 1e3),
    test_size: float = 0.3,
    random_state: int = 3,
    n_features: int | None = None,
    norm_features: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluates Linear SVM classification performance using cross-validation.

    Args:
        X (np.ndarray): Input data matrix with shape (n_samples, n_features).
        y (np.ndarray): Target labels with shape (n_samples,).
        C_range (tuple, optional): Range of regularization parameters for the SVM. Defaults to (1e-3, 1e3).
        test_size (float, optional): Proportion of the dataset to use for testing. Defaults to 0.3.
        random_state (int, optional): Seed for random number generation. Defaults to 3.
        n_features (int | None, optional): Number of features to keep after dimensionality reduction.
            If None, no reduction is performed. Defaults to None.
        verbose (bool, optional): Whether to print evaluation results. Defaults to True.

    Returns:
        Dict
    """

    best_results = {
        "acc": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "acc_std": 0.0,
        "f1_macro_std": 0.0,
        "f1_weighted_std": 0.0,
    }

    def objective(trial):
        nonlocal best_results

        sss = StratifiedShuffleSplit(
            n_splits=n_evals, test_size=test_size, random_state=random_state
        )

        svm = LinearSVC(
            C=trial.suggest_float("C", C_range[0], C_range[1], log=True),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
            dual="auto",
        )

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_scores = np.zeros(n_evals)

        # 10 times repeated holdout testing with different random splits
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            # stratified split
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            if n_features:
                # apply feature selection
                select_mask, select_idx = variational_selection(
                    X_train, y_train, n_features
                )
                X_train = X_train[:, select_mask]
                X_test = X_test[:, select_mask]

            if norm_features:
                std_scale = StandardScaler().fit(X_train)
                X_train = std_scale.transform(X_train)
                X_test = std_scale.transform(X_test)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    svm.fit(X_train, y_train)
            except ConvergenceWarning:
                # handle convergence warning
                print("ConvergenceWarning occurred during fitting.")
                break

            y_pred = svm.predict(X_test)

            accs[i] = accuracy_score(y_test, y_pred)
            f1_macros[i] = f1_score(y_test, y_pred, average="macro")
            f1_scores[i] = f1_score(y_test, y_pred, average="weighted")

            # prune trial if it is going really bad after the first half
            if (
                i > n_evals // 2
                and f1_scores[:i].mean()
                < best_results["f1_weighted"] - best_results["f1_weighted_std"]
            ):
                print("Pruning trial")
                break

        mean_f1 = f1_scores.mean()

        if mean_f1 > best_results["f1_weighted"]:
            best_results["acc"] = accs.mean()
            best_results["acc_std"] = accs.std()
            best_results["f1_macro"] = f1_macros.mean()
            best_results["f1_macro_std"] = f1_macros.std()
            best_results["f1_weighted"] = f1_scores.mean()
            best_results["f1_weighted_std"] = f1_scores.std()

        return mean_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print the mean f1 score for the best performing parameter
        print(
            f"| LIN SVM | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )
        print(f"{study.best_value=}, {study.best_params=}")

    return best_results


def svm_rbf_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 8,
    n_trials: int = 40,
    C_range: tuple = (1e-3, 1e3),
    gamma_range: tuple = (1e-3, 1e3),
    test_size: float = 0.3,
    random_state: int = 3,
    n_features: int | None = 500,
    norm_features: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluates RBF SVM classification performance using cross-validation with Optuna hyperparameter tuning.

    Args:
        X (np.ndarray): Input data matrix with shape (n_samples, n_features).
        y (np.ndarray): Target labels with shape (n_samples,).
        C_range (tuple, optional): Range of regularization parameters for the SVM. Defaults to (1e-3, 1e3).
        gamma_range (tuple, optional): Range of kernel coefficient (gamma) for the RBF kernel. Defaults to (1e-3, 1e3).
        test_size (float, optional): Proportion of the dataset to use for testing. Defaults to 0.3.
        random_state (int, optional): Seed for random number generation. Defaults to 3.
        n_features (int | None, optional): Number of features to keep after dimensionality reduction.
            If None, no reduction is performed. Defaults to None.
        verbose (bool, optional): Whether to print evaluation results. Defaults to True.

    Returns:
        Dict: Contains the mean and standard deviation of the F1 score.
    """

    best_results = {
        "acc": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "acc_std": 0.0,
        "f1_macro_std": 0.0,
        "f1_weighted_std": 0.0,
    }

    def objective(trial):
        nonlocal best_results

        sss = StratifiedShuffleSplit(
            n_splits=n_evals, test_size=test_size, random_state=random_state
        )

        svm = SVC(
            C=trial.suggest_float("C", C_range[0], C_range[1], log=True),
            gamma=trial.suggest_float(
                "gamma", gamma_range[0], gamma_range[1], log=True
            ),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
            kernel="rbf",  # Specify RBF kernel
        )

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_scores = np.zeros(n_evals)

        # 10 times repeated holdout testing with different random splits
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            # stratified split
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            if n_features:
                # apply feature selection
                select_mask, select_idx = variational_selection(
                    X_train, y_train, n_features
                )
                X_train = X_train[:, select_mask]
                X_test = X_test[:, select_mask]

            if norm_features:
                std_scale = StandardScaler().fit(X_train)
                X_train = std_scale.transform(X_train)
                X_test = std_scale.transform(X_test)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    svm.fit(X_train, y_train)
            except ConvergenceWarning:
                # handle convergence warning
                print("ConvergenceWarning occurred during fitting.")
                break

            y_pred = svm.predict(X_test)

            accs[i] = accuracy_score(y_test, y_pred)
            f1_macros[i] = f1_score(y_test, y_pred, average="macro")
            f1_scores[i] = f1_score(y_test, y_pred, average="weighted")

            # prune trial if it is going really bad after the first half
            if (
                i > n_evals // 2
                and f1_scores[:i].mean()
                < best_results["f1_weighted"] - best_results["f1_weighted_std"]
            ):
                print("Pruning trial")
                break

        mean_f1 = f1_scores.mean()

        if mean_f1 > best_results["f1_weighted"]:
            best_results["acc"] = accs.mean()
            best_results["acc_std"] = accs.std()
            best_results["f1_macro"] = f1_macros.mean()
            best_results["f1_macro_std"] = f1_macros.std()
            best_results["f1_weighted"] = f1_scores.mean()
            best_results["f1_weighted_std"] = f1_scores.std()

        return mean_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print the mean f1 score for the best performing parameter
        print(
            f"| RBF SVM | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )
        print(f"{study.best_value=}, {study.best_params=}")

    return best_results

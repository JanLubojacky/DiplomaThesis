import warnings

import numpy as np
import optuna
import optuna.logging
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from baseline_evals.feature_selection import class_variational_selection

# Before creating the study:
# optuna.logging.set_verbosity(optuna.logging.ERROR)


def svm_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 5,
    n_trials: int = 40,
    C_range: tuple = (1e-3, 1e3),
    gamma_range: tuple = (1e-3, 1e3),
    test_size: float = 0.3,
    random_state: int = 3,
    n_features_preselect: int | None = None,
    n_features: int | None = 500,
    select_n_features: bool = False,
    norm_features: bool = True,
    mode="linear",
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

        if verbose:
            print(f"Trial {trial.number + 1} / {n_trials}")

        # sss = StratifiedShuffleSplit(
        #     n_splits=n_evals, test_size=test_size, random_state=random_state
        # )
        skf = StratifiedKFold(n_splits=n_evals)

        if mode == "linear":
            params = {
                "C": trial.suggest_float("C", C_range[0], C_range[1], log=True),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                ),
                "dual": "auto",
            }
        elif mode == "rbf":
            params = {
                "C": trial.suggest_float("C", C_range[0], C_range[1], log=True),
                "gamma": trial.suggest_float(
                    "gamma", gamma_range[0], gamma_range[1], log=True
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                ),
                "kernel": "rbf",
            }

        if select_n_features:
            n_features = trial.suggest_int("n_features", 500, 5000)

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_scores = np.zeros(n_evals)

        # 10 times repeated holdout testing with different random splits
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if n_features_preselect:
                # apply feature pre-selection
                select_idx = class_variational_selection(
                    X_train, y_train, n_features_preselect
                )
                X_train = X_train[:, select_idx]
                X_test = X_test[:, select_idx]

            if norm_features:
                std_scale = StandardScaler().fit(X_train)
                X_train = std_scale.transform(X_train)
                X_test = std_scale.transform(X_test)

            if mode == "linear":
                rfe = RFE(
                    LinearSVC(**params), step=0.1, n_features_to_select=n_features
                )
            elif mode == "rbf":
                rfe = RFE(SVC(**params), step=0.1, n_features_to_select=n_features)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    rfe.fit(X_train, y_train)
            except ConvergenceWarning:
                if verbose:
                    print("ConvergenceWarning occurred during fitting.")
                break

            y_pred = rfe.predict(X_test)

            accs[i] = accuracy_score(y_test, y_pred)
            f1_macros[i] = f1_score(y_test, y_pred, average="macro")
            f1_scores[i] = f1_score(y_test, y_pred, average="weighted")

            # prune trial if it is going really bad after the first half
            if (
                i > n_evals // 2
                and f1_scores[:i].mean()
                < best_results["f1_weighted"] - best_results["f1_weighted_std"]
            ):
                if verbose:
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
            if verbose:
                print(
                    f"| SVM | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
                )

        return mean_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if mode == "linear":
        header = "LIN SVM"
    elif mode == "rbf":
        header = "RBF SVM"
    # print the mean f1 score for the best performing parameter
    print(
        f"| {header} | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
    )
    print(f"{study.best_value=}, {study.best_params=}")

    return best_results

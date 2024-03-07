import numpy as np
import optuna
import optuna.logging
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

# Before creating the study:
optuna.logging.set_verbosity(optuna.logging.ERROR)


def knn_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 20,
    n_trials: int = 20,
    nn_range: tuple = (1, 30),
    test_size: float = 0.3,
    random_state: int = 3,
    n_features: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluates k-Nearest Neighbors (kNN) classification performance using cross-validation.

    Args:
        X (np.ndarray): Input data matrix with shape (n_samples, n_features).
        y (np.ndarray): Target labels with shape (n_samples,).
        feature_num (int | None, optional): Number of features to keep after dimensionality reduction.
            If None, no reduction is performed. Defaults to None.
        verbose (bool, optional): Whether to print evaluation results. Defaults to True.

    Returns:
        Dict
    """

    best_mean = 0.0
    best_std = 0.0

    def objective(trial):
        nonlocal best_mean, best_std

        sss = StratifiedShuffleSplit(
            n_splits=n_evals, test_size=test_size, random_state=random_state
        )

        knn = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", nn_range[0], nn_range[1]),
            # weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
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

            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accs[i] = accuracy_score(y_test, y_pred)
            f1_macros[i] = f1_score(y_test, y_pred, average="macro")
            f1_scores[i] = f1_score(y_test, y_pred, average="weighted")

        mean_f1 = f1_scores.mean()

        if mean_f1 > best_mean:
            best_mean = mean_f1
            best_std = f1_scores.std()

        return mean_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print the mean f1 score for the best performing parameter
        print(f"| KNN | {best_mean:.4f} +/- {best_std:.4f} |")
        print(f"{study.best_value=}, {study.best_params=}")

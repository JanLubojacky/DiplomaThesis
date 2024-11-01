import numpy as np
import optuna
import optuna.logging
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold

# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from baseline_evals.feature_selection import class_variational_selection

# Before creating the study:
optuna.logging.set_verbosity(optuna.logging.ERROR)


def knn_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 5,
    n_trials: int = 30,
    nn_range: tuple = (1, 30),
    test_size: float = 0.3,
    random_state: int = 3,
    n_features: int = 500,
    select_n_features: bool = False,
    norm_features: bool = True,
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

    best_results = {
        "acc": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "acc_std": 0.0,
        "f1_macro_std": 0.0,
        "f1_weighted_std": 0.0,
    }

    print(n_features)

    def objective(trial):
        nonlocal best_results

        # sss = StratifiedShuffleSplit(
        #     n_splits=n_evals, test_size=test_size, random_state=random_state
        # )
        sss = StratifiedKFold(n_splits=n_evals)

        knn = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", nn_range[0], nn_range[1]),
            # weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        )

        if select_n_features:
            max = 5000
            if max > X.shape[1]:
                max = X.shape[1]
            n_features = trial.suggest_int("n_features", 50, max)

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_scores = np.zeros(n_evals)

        for i, (train_index, test_index) in enumerate(sss.split(np.zeros(len(y)), y)):
            # concat before preprocessing
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            if n_features:
                # apply feature pre-selection
                select_idx = class_variational_selection(X_train, y_train, n_features)
                X_train = X_train[:, select_idx]
                X_test = X_test[:, select_idx]

                # apply additional feature selection with mrmr

            if norm_features:
                std_scale = StandardScaler().fit(X_train)
                X_train = std_scale.transform(X_train)
                X_test = std_scale.transform(X_test)

            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accs[i] = accuracy_score(y_test, y_pred)
            f1_macros[i] = f1_score(y_test, y_pred, average="macro")
            f1_scores[i] = f1_score(y_test, y_pred, average="weighted")

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
            f"| KNN | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )
        print(f"{study.best_value=}, {study.best_params=}")


def knn_mo_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 20,
    n_trials: int = 20,
    nn_range: tuple = (1, 30),
    test_size: float = 0.3,
    random_state: int = 3,
    n_features: int | None = 500,
    norm_features: bool = True,
    verbose: bool = True,
    integration: str = None,
):
    """
    Evaluates k-Nearest Neighbors (kNN) classification performance on a multi-omic dataset using cross-validation and late integration.

    Args:
        X (np.ndarray): Input data matrix with shape (n_omics, n_samples, n_features).
        y (np.ndarray): Target labels with shape (n_samples,).
        feature_num (int | None, optional): Number of features to keep after dimensionality reduction.
            If None, no reduction is performed. Defaults to None.
        verbose (bool, optional): Whether to print evaluation results. Defaults to True.
        integration (str): how to integrate the multi-omic data, either early or late

    Returns:
        Dict
    """

    if integration == "early":
        # Transpose the matrix to make n_samples the first dimension
        X_transposed = np.transpose(X, (1, 0, 2))

        # Reshape the matrix to combine n_features and n_omics into a single dimension
        X_reshaped = X_transposed.reshape(X_transposed.shape[0], -1)

        return knn_eval(
            X_reshaped,
            y,
            n_evals,
            n_trials,
            nn_range,
            test_size,
            random_state,
            n_features,
            norm_features,
            verbose,
        )
    elif integration == "late":
        # create predictions
        y_preds = np.zeros((X.shape[0], X.shape[1]))

        for i, omic in enumerate(X):
            # train a knn model
            ...

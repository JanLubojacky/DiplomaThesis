import numpy as np
import optuna
import optuna.logging
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


def xgboost_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 5,
    n_trials: int = 100,
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

        print(f"{trial.number} / {n_trials}")

        sss = StratifiedShuffleSplit(
            n_splits=n_evals, test_size=test_size, random_state=random_state
        )

        params = {
            "verbosity": 1,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "num_class": 5,
        }

        if params["booster"] == "gbtree" or params["booster"] == "dart":
            params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )
        if params["booster"] == "dart":
            params["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            params["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        # pruning_callback = optuna.integration.XGBoostPruningCallback(
        #     trial, "validation-mlogloss"
        # )

        accs = np.zeros(n_evals)
        f1_macros = np.zeros(n_evals)
        f1_scores = np.zeros(n_evals)

        # Repeated stratified holdout testing
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            xgbst = xgb.train(
                params,
                dtrain=xgb.DMatrix(X_train, label=y_train),
                evals=[
                    (
                        xgb.DMatrix(X_test, label=y_test),
                        "validation",
                    )
                ],
                # callbacks=[pruning_callback],
                verbose_eval=False,
            )

            y_pred = xgbst.predict(xgb.DMatrix(X_test, label=y_test))

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

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print the mean f1 score for the best performing parameter
        print(
            f"| XGBoost | {best_results['acc']:.4f} +/- {best_results['acc_std']:.4f} | {best_results['f1_macro']:.4f} +/- {best_results['f1_macro_std']:.4f} | {best_results['f1_weighted']:.4f} +/- {best_results['f1_weighted_std']:.4f} |"
        )
        print(f"{study.best_value=}, {study.best_params=}")

import numpy as np
import optuna
import optuna.logging
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from baseline_evals.feature_selection import class_variational_selection


def xgboost_eval(
    X: np.ndarray,
    y: np.ndarray,
    n_evals: int = 5,
    n_trials: int = 50,
    test_size: float = 0.2,
    random_state: int = 3,
    n_features: int | None = 5000,
    norm_features: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluates XGBoost classification performance using cross-validation.

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
        # skf = StratifiedKFold(n_splits=n_evals)

        params = {
            "verbosity": 1,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "num_class": len(np.unique(y)),
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
            # randomly split test index into two halves
            val_idx, test_idx = train_test_split(
                test_index, test_size=0.5, random_state=random_state
            )

            X_train, X_val, X_test = X[train_index], X[val_idx], X[test_index]
            y_train, y_val, y_test = y[train_index], y[val_idx], y[test_index]

            if n_features:
                # apply feature selection
                select_mask, select_idx = class_variational_selection(
                    X_train, y_train, n_features
                )
                # select_mask = variance_filtering(X_train, n_features)
                X_train = X_train[:, select_mask]
                X_val = X_val[:, select_mask]
                X_test = X_test[:, select_mask]

            if norm_features:
                std_scale = StandardScaler().fit(X_train)
                X_train = std_scale.transform(X_train)
                X_val = std_scale.transform(X_val)
                X_test = std_scale.transform(X_test)

            xgbst = xgb.train(
                params,
                dtrain=xgb.DMatrix(X_train, label=y_train),
                evals=[
                    (
                        xgb.DMatrix(X_val, label=y_val),
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

        print("ACC:", accs)
        print("F1M:", f1_macros)
        print("F1W:", f1_scores)

        if mean_f1 > best_results["f1_weighted"]:
            best_results["acc"] = accs.mean()
            best_results["acc_std"] = accs.std()
            best_results["f1_macro"] = f1_macros.mean()
            best_results["f1_macro_std"] = f1_macros.std()
            best_results["f1_weighted"] = f1_scores.mean()
            best_results["f1_weighted_std"] = f1_scores.std()

            print(
                f"| XGBoost | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
            )

        return mean_f1

    study = optuna.create_study(
        direction="maximize"
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)

    if verbose:
        # print the mean f1 score for the best performing parameter
        print(
            f"| XGBoost | {best_results['acc']:.2f} +/- {best_results['acc_std']:.2f} | {best_results['f1_macro']:.2f} +/- {best_results['f1_macro_std']:.2f} | {best_results['f1_weighted']:.2f} +/- {best_results['f1_weighted_std']:.2f} |"
        )
        print(f"{study.best_value=}, {study.best_params=}")

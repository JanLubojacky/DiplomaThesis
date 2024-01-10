import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def KNN_evaluation(X, y, verbose=True):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    # # Instantiate a KNN classifier
    # knn = KNeighborsClassifier()

    # # Set up the parameter grid for GridSearchCV
    # param_grid = {"n_neighbors": np.arange(1, 20)}

    # # Define the scoring metrics
    # scoring_metrics = {
    #     "accuracy": "accuracy",
    #     "f1_macro": "f1_macro",
    #     "f1_weighted": "f1_weighted",
    # }

    # # Use GridSearchCV to find the best k
    # grid_search = GridSearchCV(
    #     knn, param_grid, scoring=scoring_metrics, cv=5, refit="f1_weighted", n_jobs=-1
    # )
    # grid_search.fit(X_train, y_train)

    # Get the results of the grid search

    for best_k in range(1, 10):
        # best_k = 3# grid_search.best_params_["n_neighbors"]

        print("Best k: ", best_k)

        # Train the model with the best k
        best_knn = KNeighborsClassifier(n_neighbors=best_k)

        # Evaluate the model using cross-validation
        # metrics = cross_validate(
        #     best_knn, X_test, y_test, cv=10, scoring=scoring_metrics, n_jobs=-1
        # )
        k = 20
        metrics = {
            "test_accuracy": np.zeros(k),
            "test_f1_macro": np.zeros(k),
            "test_f1_weighted": np.zeros(k),
        }
        for i in range(k):
            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, stratify=y
            )
            # fit the model on the training data
            best_knn.fit(X_train, y_train)
            # predict on the test data
            y_pred = best_knn.predict(X_test)
            # calculate the metrics
            metrics["test_accuracy"][i] = accuracy_score(y_test, y_pred)
            metrics["test_f1_macro"][i] = f1_score(y_test, y_pred, average="macro")
            metrics["test_f1_weighted"][i] = f1_score(y_test, y_pred, average="weighted")


        if verbose:
            # Display the results
            print("Cross Validation Results:")
            # print(
            #     f'test_accuracy: {metrics["test_accuracy"].mean():.2f} +/- {metrics["test_accuracy"].std():.2f}'
            # )
            # print(
            #     f'F1 Macro: {metrics["test_f1_macro"].mean():.2f} +/- {metrics["test_f1_macro"].std():.2f}'
            # )
            # print(
            #     f'F1 Weighted: {metrics["test_f1_weighted"].mean():.2f} +/- {metrics["test_f1_weighted"].std():.2f}'
            # )
            print(f"| KNN | {metrics['test_accuracy'].mean():.2f} ± {metrics['test_accuracy'].std():.2f} | {metrics['test_f1_macro'].mean():.2f} +/- {metrics['test_f1_macro'].std():.2f} | {metrics['test_f1_weighted'].mean():.2f} +/- {metrics['test_f1_weighted'].std():.2f}\n")
    return (
        metrics["test_f1_weighted"].mean()
        + metrics["test_accuracy"].mean()
        + metrics["test_f1_macro"].mean()
    )


def svm_crossval_feature_selection(
    X, y, k=5, kernel="linear", C=1.0, gamma=0.1, features_to_select=100, verbose=True
):
    """
    Performs feature selection using SVM and cross-validation

    Args
    ---
        X (np.array): data matrix
        y (np.array): labels
        k (int): number of folds in cross-validation
        kernel (str): kernel to use in SVM (linear, rbf)
        verbose (bool): whether to print results
    """

    if kernel == "linear":
        svm = SVC(kernel=kernel, C=C)
    elif kernel == "rbf":
        svm = SVC(kernel=kernel, C=C, gamma=gamma)

    metrics = {
        "acc": np.zeros(k),
        "f1_macro": np.zeros(k),
        "f1_weighted": np.zeros(k),
        "avg_ranks": np.zeros(X.shape[1]),
        "best_features_i": np.zeros(features_to_select),
    }
    ranks = np.zeros(X.shape[1])  # ranks of features in each fold

    for i in range(k):
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )

        # train the classifier with feature selection
        rfe = RFE(estimator=svm, n_features_to_select=features_to_select, step=1000)

        # fit the model on the training data
        rfe.fit(X_train, y_train)

        # fit the model on the training data
        svm.fit(X_train[:, rfe.support_], y_train)

        # predict on the test data
        y_pred = svm.predict(X_test[:, rfe.support_])

        # calculate the metrics
        metrics["acc"][i] = accuracy_score(y_test, y_pred)
        metrics["f1_macro"][i] = f1_score(y_test, y_pred, average="macro")
        metrics["f1_weighted"][i] = f1_score(y_test, y_pred, average="weighted")

        # save the ranks of features
        ranks += rfe.ranking_

    # average the ranks
    ranks_avg = ranks / k
    # print the best 100 feature indices based on the average ranks
    metrics["best_features_i"] = np.argsort(ranks_avg)[:features_to_select]

    if verbose:
        print(f'Accuracy: {metrics["acc"].mean():.2f} +/- {metrics["acc"].std():.2f}')
        print(
            f'F1 Macro: {metrics["f1_macro"].mean():.2f} +/- {metrics["f1_macro"].std():.2f}'
        )
        print(
            f'F1 Weighted: {metrics["f1_weighted"].mean():.2f} +/- {metrics["f1_weighted"].std():.2f}'
        )
        print(f'Best ranks: {metrics["best_features_i"]}')
        print(f'Ranks: {ranks_avg[metrics["best_features_i"]]}')

    return metrics


def svm_evaluation_linear(X, y, k=10, verbose=True, C=1.0):
    # Instantiate a SVM classifier

    metrics = {
        "acc": np.zeros(k),
        "f1_macro": np.zeros(k),
        "f1_weighted": np.zeros(k),
    }

    for i in range(k):
        # print(f"# k {i}")

        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        # Train the model with the best C
        best_svm = SVC(kernel="linear", probability=True, C=C)

        # fit the model on the training data
        best_svm.fit(X_train, y_train)

        # predict on the test data
        y_pred = best_svm.predict(X_test)

        # calculate the metrics
        metrics["acc"][i] = accuracy_score(y_test, y_pred)
        metrics["f1_macro"][i] = f1_score(y_test, y_pred, average="macro")
        metrics["f1_weighted"][i] = f1_score(y_test, y_pred, average="weighted")

    # Display the results
    print("\nCross Validation Results:")
    print(f'Accuracy: {metrics["acc"].mean():.2f} +/- {metrics["acc"].std():.2f}')
    print(
        f'F1 Macro: {metrics["f1_macro"].mean():.2f} +/- {metrics["f1_macro"].std():.2f}'
    )
    print(
        f'F1 Weighted: {metrics["f1_weighted"].mean():.2f} +/- {metrics["f1_weighted"].std():.2f}'
    )

    return (
        metrics["f1_weighted"].mean()
        + metrics["acc"].mean()
        + metrics["f1_macro"].mean()
    )


def XGBoost_evaluation(X, y, best_params, k=10, verbose=True):
    # Define the scoring metrics
    print("Evaluating model...")
    if verbose:
        print("Best params: ", best_params)
    # Train the model with the best parameters
    best_xgb = XGBClassifier(
        objective="multi:softmax", num_class=len(np.unique(y)), **best_params
    )

    metrics = {
        "acc": np.zeros(k),
        "f1_macro": np.zeros(k),
        "f1_weighted": np.zeros(k),
    }

    for i in range(k):
        # split data into train and test, the test size is higher
        # to have a reasonable number of samples
        # in each fold
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y
        )

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        # fit the model on the training data
        # pass the sample weights to the fit method!
        best_xgb.fit(X_train, y_train, sample_weight=sample_weights)
        # predict on the test data
        y_pred = best_xgb.predict(X_test)

        # interface 1
        # bst = xgb.train(
        #     best_params,
        #     dtrain=xgb.DMatrix(X_train, label=y_train, weight=sample_weights),
        #     verbose_eval=False,
        # )
        # preds = bst.predict(xgb.DMatrix(X_test))
        # y_pred = np.rint(preds)

        # calculate the metrics
        metrics["acc"][i] = accuracy_score(y_test, y_pred)
        metrics["f1_macro"][i] = f1_score(y_test, y_pred, average="macro")
        metrics["f1_weighted"][i] = f1_score(y_test, y_pred, average="weighted")

    # Display the results
    print("\nCross Validation Results:")
    print(f'Accuracy: {metrics["acc"].mean():.2f} +/- {metrics["acc"].std():.2f}')
    print(
        f'F1 Macro: {metrics["f1_macro"].mean():.2f} +/- {metrics["f1_macro"].std():.2f}'
    )
    print(
        f'F1 Weighted: {metrics["f1_weighted"].mean():.2f} +/- {metrics["f1_weighted"].std():.2f}'
    )

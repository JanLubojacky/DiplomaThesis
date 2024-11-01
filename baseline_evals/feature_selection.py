import numpy as np


def variance_filtering(X_train, n_features=500, count_th=0.9):
    """
    Order features by variance and select the ones with the biggest variance

    Args:
        X_train: np.array of shape (n_samples, n_features)
        n_features: int, number of features to select
    Returns:
        best_mask: np.array of shape (n_features,) with True for selected features
    """

    # ascending order of variance
    var_order = np.argsort(X_train.var(axis=0))

    # select the features with the largest variance
    best_indices = var_order[-n_features:]

    return best_indices


# def count_filtering(X_train, count_th):
#     """
#     Order features by count and select the ones with the biggest count
#
#     Args:
#         X_train: np.array of shape (n_samples, n_features)
#     Returns:
#         best_mask: np.array of shape (n_features,) with True for selected features
#     """
#
#     # ascending order of count
#     count_order = np.argsort(X_train.sum(axis=0))
#
#     # filter out bottom count_th features
#     best_mask = count_order <= (X_train.shape[1] - int(count_th * X_train.shape[1]))
#
#     print((X_train.shape[1] - int(count_th * X_train.shape[1])))
#
#     return best_mask


def class_variational_selection(X_train, y, n_features=500):
    """
    Given training data X_train (n_samples, n_features) and labels y (n_samples,)
    filter the features by the variance of the feature mean accross the classes
    """
    # select features based on the training set
    num_labels = len(np.unique(y))
    features_class_mean = np.zeros((X_train.shape[1], num_labels))

    # separate training samples by class
    for label in range(num_labels):
        X_class = X_train[y == label]
        features_class_mean[:, label] = X_class.mean(axis=0)

    feat_class_vars = features_class_mean.var(axis=1)
    class_var_order = np.argsort(feat_class_vars)

    # select the features with the largest variance
    # best_mask = class_var_order >= (feat_class_vars.shape[0] - n_features)
    # best_idx = np.where(best_mask)[0]
    best_idx = class_var_order[-n_features:]

    return best_idx

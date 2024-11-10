import warnings

import numpy as np
import statsmodels.api as sm
import torch


def cosine_similarity_matrix(matrix):
    """
    Given a matrix of (n_samples, n_features) compute the cosine similarities, between the samples
    """
    # Compute dot product between all pairs of vectors
    dot_products = torch.matmul(matrix, matrix.T)

    # Compute magnitudes of all vectors
    magnitudes = torch.norm(matrix, dim=1)

    # Compute outer product of magnitudes to obtain matrix of magnitudes product
    magnitude_products = magnitudes.unsqueeze(0) * magnitudes.unsqueeze(1)

    # Compute cosine similarity matrix
    cosine_similarities = dot_products / magnitude_products

    return cosine_similarities


def threshold_matrix(
    cosine_similarities,
    self_connections,
    target_avg_degree=None,
    avg_degree_tol=0.5,
    th_min=0.0,
    th_max=1.0,
    verbose=False,
    self_loops=True,
):
    """
    Given a matrix of cosine similarities, threshold the values to obtain a binary adjacency matrix
    if target_avg_degree is given a binary search is used to search for a threshold that will result in
    target_avg_degree +/- avg_degree_tol

    Args:
        cosine_similarities (torch.Tensor): A 2D tensor representing the cosine similarities between samples
        self_connections (bool): Whether to allow self-connections in the graph
        target_avg_degree (float): The target average degree for the graph
        avg_degree_tol (float): The tolerance for the average degree
        th_min (float): The minimum threshold value for binarisation
        th_max (float): The maximum threshold value for binarisation
        verbose (bool): Whether to print the number of isolated samples and the average degree
    """

    # Perform binary search to find the optimal threshold
    while th_max - th_min > 1e-6:
        th = (th_min + th_max) / 2
        A = torch.where(cosine_similarities > th, 1.0, 0.0)

        if not self_connections:
            A = A - torch.eye(A.shape[0], dtype=torch.float32)

        current_degree = A.sum(dim=1).mean()

        if target_avg_degree is not None:
            if current_degree > target_avg_degree - avg_degree_tol:
                th_min = th
            elif current_degree < target_avg_degree + avg_degree_tol:
                th_max = th

            if (
                target_avg_degree - avg_degree_tol
                < current_degree
                < target_avg_degree + avg_degree_tol
            ):
                break
        else:
            break

    if verbose:
        min_degree = 0
        if self_loops:
            min_degree = 1
        print(
            f"Isolated samples = {(A.sum(dim=1) == min_degree).sum()}, avg degree = {current_degree}"
        )

    return A


def keep_n_neighbours(cosine_similarity, n, self_connections=False):
    """
    Keep only the n highest values in each row of a matrix, setting all other values to 0
    """
    rows, cols = cosine_similarity.shape
    if self_connections:
        A = torch.eye(rows, dtype=torch.float32)
    else:
        A = torch.zeros_like(cosine_similarity, dtype=torch.float32)

    top_k_indices = torch.topk(cosine_similarity, n, largest=True, dim=1).indices

    for r in range(rows):
        A[r, top_k_indices[r]] = 1

    return A


def dense_to_coo(adj_mat):
    """
    Convert an adjacency matrix in a dense format to a torch tensor in COO format
    """

    # mask upper triangular part of the matrix
    # adj_mat = torch.triu(adj_mat, diagonal=1)

    indices = torch.nonzero(adj_mat, as_tuple=True)

    return torch.stack(indices, dim=0)


def dense_to_attributes(adj_mat):
    """
    return the flattened weight entries from the matrix
    for use as edge attributes
    """
    return adj_mat[adj_mat != 0].view(-1, 1)


def create_diff_exp_connections_norm(X, multiplier=1.0):
    """
    This function identifies and categorizes gene expression levels as
    under-expressed (-1), over-expressed (1), or baseline (0) based on standard
    deviation from the mean.

    Args:
        X (torch.Tensor): A 2D PyTorch tensor representing gene expression data,
            with shape (samples, genes).
        std_multiplier (float, optional): A hyperparameter that scales the
            standard deviation to define the expression bounds. Higher values will result
            in stricter bounds (std_multiplier=2.0 will consider values further than 2 sigma
            from the mean to be differential) Defaults to 1.0.

    Returns:
        torch.Tensor: A 2D PyTorch tensor with the same shape as X, where each element
            indicates the expression category (-1, 0, or 1) for the corresponding
            gene in each sample.
    """

    # fit the differential expression model
    mean_exps = X.mean(dim=0)
    exps_std = X.std(dim=0)

    print(mean_exps.shape, exps_std.shape)

    lb_exps = mean_exps - exps_std * multiplier
    ub_exps = mean_exps + exps_std * multiplier

    A_exps = torch.zeros_like(X)

    mask_below = X <= lb_exps
    mask_above = X >= ub_exps

    A_exps[mask_below] = -1  # Set under-expressed elements
    A_exps[mask_above] = 1  # Set over-expressed elements

    print("isolated sample nodes, isolated gene nodes, mean degree: ")
    print(
        (A_exps.abs().sum(axis=1) == 0).sum(),
        (A_exps.abs().sum(axis=0) == 0).sum(),
        A_exps.abs().sum() / A_exps.shape[0],
    )

    return A_exps


def diff_exp_connections_nbnom(expression_vector, var_multiplier=1):
    """
    Estimate the differential expression of a gene using the negative binomial distribution.

    Args:
        expression_vector (np.ndarray): The expression vector of the gene.
        var_multiplier (int): The multiplier for the variance threshold. Default is 1.
    Returns:
        select_mask (np.ndarray): The mask of the selected samples.
    """
    if not isinstance(expression_vector, np.ndarray):
        expression_vector = np.array(expression_vector)

    # ignore all warnings
    with warnings.catch_warnings():
        # for some distributions, the fitting will fail, so ignore warnings for those
        warnings.filterwarnings("ignore")
        res = sm.NegativeBinomial(expression_vector, np.ones_like(expression_vector)).fit(
            start_params=[1, 1], disp=0
        )

    mu = np.exp(res.params[0])
    p = 1 / (1 + mu * res.params[1])
    r = mu * p / (1 - p)

    var = r * (1 - p) / p**2
    # var = np.sqrt(var)
    # std = var # np.sqrt(var)
    mask_above = expression_vector > mu + var * var_multiplier
    mask_below = expression_vector < mu - var * var_multiplier

    return mask_below, mask_above


def create_diff_exp_connections_nbnom(X, train_mask, var_multiplier=1.0):
    """
    This function identifies and categorizes gene expression levels as
    under-expressed (-1), over-expressed (1), or baseline (0) based on the negative binomial distribution.

    Args:
        X (torch.Tensor): A 2D PyTorch tensor representing gene expression data,
            with shape (samples, genes).
        var_multiplier (float, optional): A hyperparameter that scales the
            standard deviation to define the expression bounds. Higher values will result
            in stricter bounds (std_multiplier=2.0 will consider values further than 2 sigma
            from the mean to be differential) Defaults to 1.0.
    Returns:
        A (torch.Tensor): A 2D PyTorch tensor with the same shape as X, where each element
            indicates the expression category (-1, 0, or 1) for the corresponding
            gene in each sample.
        isolated_nodes_mask (torch.Tensor): A 1D PyTorch tensor indicating the indices of
            genes that are not differentially expressed.
    """

    A = torch.zeros_like(X)

    # X = X[train_mask]

    # print(A.shape, X.shape)

    for i in range(X.shape[1]):
        mask_below, mask_above = diff_exp_connections_nbnom(X[:, i], var_multiplier)
        A[mask_below, i] = -1
        A[mask_above, i] = 1

    # isolated_nodes_mask = torch.sum(torch.abs(A), dim=1) == 0
    # A = A[~isolated_nodes_mask]
    print("isolated sample nodes, isolated gene nodes, mean degree: ")
    print(
        (A.abs().sum(axis=1) == 0).sum(),
        (A.abs().sum(axis=0) == 0).sum(),
        A.abs().sum() / A.shape[0],
    )

    return A  # , isolated_nodes_mask

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


def keep_n_neighbours(A, n):
    """
    Keep only the n highest values in each row of a matrix, setting all other values to 0
    """
    rows, cols = A.shape
    for i in range(rows):
        bottom_k_indices = torch.topk(A[i], n, largest=False).indices
        A[i][bottom_k_indices] = 0
    return A


def dense_to_coo(adj_mat):
    """
    Convert an adjacency matrix in a dense format to a torch tensor in COO format
    """
    indices = torch.nonzero(adj_mat, as_tuple=True)
    return torch.stack(indices, dim=0)


def dense_to_attributes(adj_mat):
    """
    return the flattened weight entries from the matrix
    for use as edge attributes
    """
    return torch.tensor(adj_mat[adj_mat != 0]).view(-1, 1)


def create_expression_connections(X, std_multiplier=1.0):
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
    mean_exps = X.mean(dim=0)
    exps_std = X.std(dim=0)

    lb_exps = mean_exps - exps_std * std_multiplier
    ub_exps = mean_exps + exps_std * std_multiplier

    A_exps = torch.zeros_like(X)

    mask_below = X <= lb_exps
    mask_above = X >= ub_exps

    A_exps[mask_below] = -1  # Set under-expressed elements
    A_exps[mask_above] = 1  # Set over-expressed elements

    return A_exps

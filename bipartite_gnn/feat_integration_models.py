import torch
import torch.nn.functional as F
import torch_geometric as pyg

import torch.nn as nn


class LinearIntegration(torch.nn.Module):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_samples, n_features)
    Returns
        predictions for each class
    """

    def __init__(self, n_views, view_dim, n_classes, hidden_dim, dropout=0.2):
        super().__init__()
        self.lin1 = pyg.nn.Linear(-1, hidden_dim, weight_initializer="glorot")
        self.lin2 = pyg.nn.Linear(-1, n_classes, weight_initializer="glorot")
        self.dropout = dropout

    def forward(self, x):
        """
        where x is (n_omics, n_samples, n_features)
        """

        # (n_omics, n_samples, n_features) -> (n_samples, n_omics, n_features)
        xt = torch.transpose(x, 0, 1)
        # (n_samples, n_omics, n_features) -> (n_samples, n_features*n_omics)
        x = xt.reshape(xt.shape[0], -1)

        x = self.lin1(x)
        x = F.elu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin2(x)


class AttentionIntegrator(torch.nn.Module):
    """
    Integrates multi-omics data using a self-attention mechanism.
    """

    def __init__(
        self,
        n_views,
        view_dim,
        n_classes,
        hidden_dim,
        dropout=0.2,
        one_lin_layer=False,
        elu_alpha=1.0,
    ):
        super().__init__()
        self.n_views = n_views
        self.view_dim = view_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.one_lin_layer = one_lin_layer
        self.elu_alpha = elu_alpha

        # Linear layers for the self-attention mechanism
        self.query = pyg.nn.Linear(-1, hidden_dim, weight_initializer="glorot")
        self.key = pyg.nn.Linear(-1, hidden_dim, weight_initializer="glorot")
        self.value = pyg.nn.Linear(-1, hidden_dim, weight_initializer="glorot")

        # Linear layers for the final transformation
        self.lin1 = pyg.nn.Linear(-1, hidden_dim, weight_initializer="glorot")
        self.lin2 = pyg.nn.Linear(-1, n_classes, weight_initializer="glorot")

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (n_omics, n_samples, n_features) -> (n_samples, n_omics, n_features)
        xt = torch.transpose(x, 0, 1)

        # query, key, and value matrices
        q = self.query(xt)
        k = self.key(xt)
        v = self.value(xt)

        # Compute the scaled dot-product attention
        qkt = torch.matmul(q, k.transpose(1, 2))
        qkt = F.softmax(qkt, dim=-1) / (self.hidden_dim**0.5)

        x = torch.matmul(qkt, v)

        if self.one_lin_layer:
            x = self.lin1(x)
            x = F.elu(x, alpha=self.elu_alpha)

        # (n_samples, n_omics, n_features) -> (n_samples, n_features*n_omics)
        x = x.reshape(x.shape[0], -1)

        x = self.lin2(x)

        return x


def reshape_tensor(tensor):
    """
    Given a tensor of shape (n^m, n), reshape it to (n, n, ..., n) where n is the last dimension of the tensor
    """
    # Step 1: Determine the shape of the input tensor
    shape = tensor.shape[0]
    out_dim = tensor.shape[1]

    if len(shape) == out_dim:
        return tensor

    # Step 2: Calculate the number of dimensions needed
    num_dims = 0
    while shape % out_dim == 0:
        shape = shape // out_dim
        num_dims += 1

    # Step 3: Reshape the tensor
    # The new shape is (n,) repeated num_dims times, followed by the original shape from the second dimension onwards
    new_shape = (out_dim,) * (num_dims + 1)

    return tensor.view(*new_shape)


def construct_cross_feature_discovery_tensor_single(x, flatten_output=True):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_features)
    """

    # if num of omics is 1, return the tensor
    # and remove the first dimension
    if x.shape[0] == 1:
        return x.squeeze()

    # Initialize the output tensor with the first 2D tensor
    output_tensor = torch.outer(x[0], x[1])

    # Iterate over the remaining tensors in the list
    for tensor in x[2:]:
        # Flatten the output tensor
        flattened_output = torch.flatten(output_tensor)
        # Compute the outer product with the current tensor
        output_tensor = torch.outer(flattened_output, tensor)
    if flatten_output:
        return torch.flatten(output_tensor)

    return reshape_tensor(output_tensor)


def construct_cross_feature_discovery_tensor(x, flatten_output=True):
    """
    Constructs a tensor representing the cross-feature interactions between multiple samples.

    Args:
        x: torch.Tensor, where x is (n_samples, n_omics, n_features)
        flatten_output: bool, whether to flatten the output tensor (default: True)

    Returns:
        torch.Tensor, the cross-feature discovery tensor
    """

    # (n_omics, n_samples, n_features) -> (n_samples, n_omics, n_features)
    # x = torch.transpose(x, 0, 1)

    # Get the number of samples, omics, and features
    n_samples, n_omics, n_features = x.shape

    # Initialize a list to store the output tensors for each sample
    output_tensors = torch.zeros(n_samples, n_features**n_omics)

    # Iterate over each sample
    for sample_idx in range(n_samples):
        output_tensors[sample_idx] = construct_cross_feature_discovery_tensor_single(
            x[sample_idx]
        )

    return output_tensors


class VCDN(torch.nn.Module):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_samples, n_features)
    Returns
        predictions for each class
    """

    def __init__(self, n_views, view_dim, n_classes, hidden_dim, convolutional=False):
        super().__init__()
        self.convolutional = convolutional

        if convolutional:
            raise NotImplementedError
            # TODO
            # use a convolutional layer to classify the samples, kernel dim is (view_dim, ) * n_views
            self.classifier = torch.nn.Conv2d(
                n_views, n_classes, kernel_size=(1, view_dim)
            )
        else:
            self.lin1 = pyg.nn.Linear(-1, hidden_dim, weight_initializer="glorot")
            # use a single linear layer to classify the samples
            self.classifier = pyg.nn.Linear(
                hidden_dim, n_classes, weight_initializer="glorot"
            )

    def forward(self, x):
        """
        where x is (n_samples, n_omics, n_features)
        """

        # print(x.shape)

        if self.convolutional:
            cfdt = construct_cross_feature_discovery_tensor(x, flatten_output=False)
        else:
            cfdt = construct_cross_feature_discovery_tensor(x)

        x = self.lin1(cfdt)
        x = F.elu(x)

        return self.classifier(x)

import torch
import torch.nn.functional as F
import torch_geometric as pyg


class LinearIntegration(torch.nn.Module):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_samples, n_features)
    Returns
        predictions for each class
    """

    def __init__(self, n_views, view_dim, n_classes, hidden_dim, dropout=0.5):
        super().__init__()
        self.lin1 = pyg.nn.Linear(-1, hidden_dim, weight_initializer="kaiming_uniform")
        self.lin2 = pyg.nn.Linear(
            hidden_dim, n_classes, weight_initializer="kaiming_uniform"
        )
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
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin2(x)


class AttentionIntegration(torch.nn.Module):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_samples, n_features)
    Returns
        predictions for each class
    """

    def __init__(self, n_views, view_dim, n_classes):
        super(AttentionIntegration, self).__init__()
        self.n_views = n_views
        self.view_dim = view_dim
        self.n_classes = n_classes

        # Attention mechanism
        self.attention = torch.nn.Sequential(
            pyg.nn.Linear(view_dim, view_dim // 2),
            torch.nn.ReLU(),
            pyg.nn.Linear(view_dim // 2, 1),
        )

        # Classifier
        self.classifier = pyg.nn.Linear(view_dim * n_views, n_classes)

    def forward(self, x):
        """
        where x is (n_omics, n_samples, n_features)
        """
        # Calculate attention weights for each omic view
        attention_weights = torch.softmax(self.attention(x), dim=0)

        # Apply attention weights to each omic view
        weighted_views = x * attention_weights

        # Integrate information from all views
        integrated_views = torch.sum(weighted_views, dim=0)

        # Flatten the integrated views
        integrated_views_flat = integrated_views.view(integrated_views.size(0), -1)

        # Pass through the classifier
        predictions = self.classifier(integrated_views_flat)

        return F.relu(predictions)


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


def construct_cross_feature_discovery_tensor(x, flatten_output=True):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_samples, n_features)
    """

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


class VCDN(torch.nn.Module):
    """
    Args
        x: torch.Tensor, where x is (n_omics, n_samples, n_features)
    Returns
        predictions for each class
    """

    def __init__(self, n_views, view_dim, n_classes, convolutional=False):
        super(VCDN, self).__init__()
        self.convolutional = convolutional

        if convolutional:
            raise NotImplementedError
            # TODO
            # use a convolutional layer to classify the samples, kernel dim is (view_dim, ) * n_views
            self.classifier = torch.nn.Conv2d(
                n_views, n_classes, kernel_size=(1, view_dim)
            )
        else:
            # use a single linear layer to classify the samples
            self.classifier = pyg.nn.Linear(
                -1, n_classes, weight_initializer="kaiming_uniform"
            )

    def forward(self, x):
        """
        where x is (n_omics, n_samples, n_features)
        """
        if self.convolutional:
            cfdt = construct_cross_feature_discovery_tensor(x, flatten_output=False)
        else:
            cfdt = construct_cross_feature_discovery_tensor(x)

        return F.relu(self.classifier(cfdt))

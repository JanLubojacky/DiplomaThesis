import torch
import torch.nn.functional as F
import torch_geometric as pyg

from src.gnn_utils.feat_integration_modules import (
    LinearIntegration,
    AttentionIntegrator,
    VCDN,
)


class BiRGAT(torch.nn.Module):
    def __init__(
        self,
        omic_channels,
        feature_names,
        relations,
        input_dims,
        num_classes,
        hidden_channels,
        heads,
        dropout,
        attention_dropout,
        integrator_type="linear",
        three_layers=False,
        seed=12345,
    ) -> None:
        torch.manual_seed(seed)
        super().__init__()

        self.omic_channels = omic_channels
        self.heads = heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.three_layers = three_layers

        all_names = omic_channels + feature_names

        self.conv1 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    hidden_channels[0],
                    hidden_channels[1],
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    dropout=attention_dropout,
                    edge_dim=1,
                )
                for relation in relations
            }
        )
        self.self_loops1 = {
            name: pyg.nn.Linear(
                hidden_channels[0],
                hidden_channels[1],
                weight_initializer="kaiming_uniform",
            )
            for name in all_names
        }

        self.conv2 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    hidden_channels[1] * heads,
                    hidden_channels[2],
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    dropout=dropout,
                    edge_dim=1,
                )
                for relation in relations
            }
        )
        self.self_loops2 = {
            name: pyg.nn.Linear(
                hidden_channels[1] * heads,
                hidden_channels[2],
                weight_initializer="kaiming_uniform",
            )
            for name in all_names
        }

        self.conv3 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    hidden_channels[2] * heads,
                    hidden_channels[3],
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                    dropout=dropout,
                    edge_dim=1,
                )
                for relation in relations
            }
        )
        self.self_loops3 = {
            name: pyg.nn.Linear(
                hidden_channels[2] * heads,
                hidden_channels[3],
                weight_initializer="kaiming_uniform",
            )
            for name in all_names
        }

        if integrator_type == "linear":
            self.integrator = LinearIntegration(
                n_views=len(omic_channels),
                view_dim=hidden_channels[3],
                n_classes=num_classes,
                hidden_dim=hidden_channels[4],
            )
        elif integrator_type == "attention":
            self.integrator = AttentionIntegrator(
                n_views=len(omic_channels),
                view_dim=hidden_channels[3],
                n_classes=num_classes,
                hidden_dim=hidden_channels[4],
            )
        elif integrator_type == "vcdn":
            self.integrator = VCDN(
                n_views=len(omic_channels),
                view_dim=hidden_channels[3],
                n_classes=num_classes,
                hidden_dim=hidden_channels[4],
            )
        elif integrator_type is None:
            self.integrator = None
        else:
            raise ValueError(
                "Invalid integrator type, please choose from 'linear', 'attention', 'vcdn'"
            )

    def forward(self, data):
        """
        Returns the logits for the sample nodes

        Args:
            data: HeteroData

        Returns:
            x (n_samples, n_classes): torch.Tensor
        """

        x_dict = data.x_dict
        edge_dict = data.edge_index_dict

        x1 = self.conv1(x_dict, edge_dict)
        for key in self.omic_channels:
            x1[key] = x1[key] + self.self_loops1[key](x_dict[key]).repeat(1, self.heads)
            x1[key] = F.dropout(x1[key], p=self.dropout, training=self.training)
        x1 = {key: F.relu(x1[key]) for key in x_dict.keys()}

        if self.three_layers:
            x2 = self.conv2(x1, edge_dict)
            for key in self.omic_channels:
                x2[key] = x2[key] + self.self_loops2[key](x1[key]).repeat(1, self.heads)
                x2[key] = F.dropout(x2[key], p=self.dropout, training=self.training)
            x2 = {key: F.relu(x2[key]) for key in x_dict.keys()}
        else:
            # skip second layer
            x2 = x1

        x3 = self.conv3(x2, edge_dict)
        for key in self.omic_channels:
            x3[key] = x3[key] + self.self_loops3[key](x2[key])
            x3[key] = F.dropout(x3[key], p=self.dropout, training=self.training)
        x3 = {key: F.relu(x3[key]) for key in x_dict.keys()}

        x_stack = []
        for omic in self.omic_channels:
            x_stack.append(x3[omic])
        # stack omic layers into (n_omics, n_samples, n_features)
        x_sample_features = torch.stack(x_stack)
        # reorder to (n_samples, n_omics, n_features)
        x_sample_features = torch.transpose(x_sample_features, 0, 1)

        if self.integrator is not None:
            x = self.integrator(x_sample_features)
        else:
            x = x_sample_features.squeeze()

        return x

    def get_feature_importances(self, data, feature_names):
        """
        Calculates feature importances using a feature ablation approach.

        Args:
            data (Data): PyTorch Geometric Data object containing the input features and labels.
            feature_names (dict): A dictionary with keys as omics and values as lists of feature names.

        Returns:
            feature_importances (dict): A dictionary with keys as omics and values as tensors of feature importances.
        """
        base_loss = torch.nn.CrossEntropyLoss()(
            self.forward(data)[data.test_mask], data.y[data.test_mask]
        )

        print("base_loss", base_loss)

        feature_importances = {}

        for omic in data.x_dict.keys():
            feature_importances[omic] = torch.zeros(data.x_dict[omic].shape[1])

            for i, feature_name in enumerate(feature_names[omic]):
                temp_feature = data.x_dict[omic][:, i].clone()
                data.x_dict[omic][:, i] = 0
                y_hat = self.forward(data)
                loss = torch.nn.CrossEntropyLoss()(y_hat, data.y)
                feature_importances[omic][i] = loss - base_loss
                data.x_dict[omic][:, i] = temp_feature

            sort_idx = torch.argsort(feature_importances[omic], descending=True)
            feature_names[omic] = [feature_names[omic][i] for i in sort_idx]
            feature_importances[omic] = feature_importances[omic][sort_idx]

            print(f"omic: {omic}")
            print(f"feature_names: {feature_names[omic]}")
            print(f"feature_importances: {feature_importances[omic]}")

        return feature_names, feature_importances

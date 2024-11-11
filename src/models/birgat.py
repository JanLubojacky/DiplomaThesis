import torch
import torch.functional as F
import torch_geometric as pyg

from src.gnn_utils.feat_integration_modules import (
    VCDN,
    AttentionIntegrator,
    LinearIntegration,
)

class BiRGAT(torch.nn.Module):
    def __init__(
        self,
        omic_channels,
        feature_names,
        relations,
        input_dims,
        num_classes,
        proj_dim,
        hidden_channels,
        heads,
        dropout,
    ) -> None:
        super().__init__()

        self.omic_channels = omic_channels
        self.heads = heads
        self.dropout = dropout

        all_names = omic_channels + feature_names

        self.projections = {
            omic: pyg.nn.Linear(input_dim, proj_dim, weight_initializer="glorot")
            for omic, input_dim in zip(omic_channels, input_dims)
        }

        self.conv1 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    proj_dim,
                    hidden_channels[0],
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    dropout=dropout,
                    edge_dim=1,
                )
                for relation in relations
            }
        )
        self.self_loops1 = {
            name: pyg.nn.Linear(
                proj_dim, hidden_channels[0], weight_initializer="glorot"
            )
            for name in all_names
        }

        self.conv2 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    hidden_channels[0] * heads,
                    hidden_channels[1],
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
                hidden_channels[0] * heads,
                hidden_channels[1],
                weight_initializer="glorot",
            )
            for name in all_names
        }

        self.conv3 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    hidden_channels[1] * heads,
                    hidden_channels[2],
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
                hidden_channels[1] * heads,
                hidden_channels[2],
                weight_initializer="glorot",
            )
            for name in all_names
        }

        self.skip_connection = {
            name: pyg.nn.Linear(
                proj_dim,
                hidden_channels[2],
                weight_initializer="glorot",
            )
            for name in all_names
        }

        # self.integrator = LinearIntegration(
        #     n_views=len(omic_channels),
        #     view_dim=hidden_channels[2],
        #     n_classes=num_classes,
        #     hidden_dim=hidden_channels[3],
        # )
        self.integrator = AttentionIntegrator(
            n_views=len(omic_channels),
            view_dim=hidden_channels[2],
            n_classes=num_classes,
            hidden_dim=hidden_channels[3],
            one_lin_layer=False,
        )
        # self.integrator = VCDN(
        #     n_views=len(omic_channels),
        #     view_dim=hidden_channels[2],
        #     n_classes=num_classes,
        #     hidden_dim=hidden_channels[3],
        # )

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

        for omic in self.omic_channels:
            x_dict[omic] = F.elu(self.projections[omic](x_dict[omic]))
            # x_dict[omic] = F.dropout(
            #     x_dict[omic], p=self.dropout, training=self.training
            # )

        # print(x_dict[omic].shape)
        #
        # print(x_dict)
        # print(edge_dict)

        x1 = self.conv1(x_dict, edge_dict)
        for key in self.omic_channels:
            x1[key] = x1[key] + self.self_loops1[key](x_dict[key]).repeat(1, self.heads)
        x1 = {key: F.elu(x1[key]) for key in x_dict.keys()}

        x2 = self.conv2(x1, edge_dict)
        for key in self.omic_channels:
            x2[key] = x2[key] + self.self_loops2[key](x1[key]).repeat(1, self.heads)
        x2 = {key: F.elu(x2[key]) for key in x_dict.keys()}

        x3 = self.conv3(x2, edge_dict)
        for key in self.omic_channels:
            x3[key] = x3[key] + self.self_loops3[key](x2[key])
        x3 = {key: F.elu(x3[key]) for key in x_dict.keys()}

        # skip connection
        # for key in self.omic_channels:
        #     x3[key] = x3[key] + self.self_loops1[key](
        #         x_dict[key]
        #     )  # .repeat(1, self.heads)

        # x3 = {
        #     key: F.dropout(x3[key], p=0.2, training=self.training)
        #     for key in self.omic_channels
        # }

        x_stack = []
        for omic in self.omic_channels:
            x_stack.append(x3[omic])
        x_sample_features = torch.stack(x_stack)

        return self.integrator(x_sample_features)

    def projection_layers_l1_norm(self):
        """
        Returns the l1 norm of the projection layers
        """
        l1_norm = torch.tensor(0.0)
        for proj_layer in self.projections.values():
            l1_norm += torch.norm(proj_layer.weight.to("cpu"), p=1)

        return l1_norm

    def proj_layer_feature_importance(self):
        """
        Returns the feature importance for each projection layer,
        the layer is sparse regylarized, so we sum the absolute values of the weights
        given to each feature
        """

        feature_importances = {}

        for omic, proj_layer in self.projections.items():
            feature_importances[omic] = proj_layer.weight.abs().sum(dim=0)

        return feature_importances

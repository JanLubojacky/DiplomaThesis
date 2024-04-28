import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GATv2Conv

# from bipartite_gnn.feat_integration_models import LinearIntegration
from bipartite_gnn.feat_integration_models import AttentionIntegrator
# from bipartite_gnn.feat_integration_models import VCDN


def create_HeteroConv(
    relations, edge_dims, dropout, in_channels, out_channels, heads, concat
):
    conv_dict = {}
    for relation in relations:
        if edge_dims.get(relation) is not None:
            conv_dict[relation] = pyg.nn.GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                add_self_loops=False,
                dropout=dropout,
                edge_dim=edge_dims[relation],
            )
        else:
            conv_dict[relation] = pyg.nn.GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                add_self_loops=False,
                dropout=dropout,
            )
    return pyg.nn.HeteroConv(conv_dict)


class NN_encoders_ATT_integrator(torch.nn.Module):
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
        self.dropout = dropout

        self.projections = {
            omic: pyg.nn.Linear(input_dim, proj_dim, weight_initializer="glorot")
            for omic, input_dim in zip(omic_channels, input_dims)
        }

        self.lin2s = {
            omic: pyg.nn.Linear(proj_dim, num_classes, weight_initializer="glorot")
            for omic, input_dim in zip(omic_channels, input_dims)
        }

        # self.integrator = AttentionIntegrator(
        #     n_views=len(omic_channels),
        #     view_dim=hidden_channels[0],
        #     n_classes=num_classes,
        #     hidden_dim=hidden_channels[1],
        # )

    def forward(self, data):
        x_dict = data.x_dict

        for omic in self.omic_channels:
            x_dict[omic] = F.elu(self.projections[omic](x_dict[omic]))
            F.dropout(x_dict[omic], p=self.dropout, training=self.training)

        for omic in self.omic_channels:
            x_dict[omic] = F.elu(self.lin2s[omic](x_dict[omic]))
            F.dropout(x_dict[omic], p=self.dropout, training=self.training)

        x_stack = []
        for omic in self.omic_channels:
            x_stack.append(x_dict[omic])
        x_sample_features = torch.stack(x_stack)

        # return self.integrator(x_sample_features)

    def projection_layers_l1_norm(self):
        """
        Returns the l1 norm of the projection layers
        """
        l1_norm = torch.tensor(0.0)
        for proj_layer in self.projections.values():
            l1_norm += torch.norm(proj_layer.weight.to("cpu"), p=1)
        return l1_norm


class GAT_2L(torch.nn.Module):
    # """"""
    def __init__(
        self,
        input_shape,
        num_labels,
        proj_dim,
        hidden_channels,
        num_heads,
        dropout=0.1,
        eps=0.9,
    ):
        super().__init__()
        torch.manual_seed(1234567)
        self.projections = pyg.nn.Linear(input_shape, proj_dim)
        self.eps = eps

        self.conv1 = GATv2Conv(
            in_channels=proj_dim,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False,
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False,
        )
        self.lin1 = pyg.nn.Linear(hidden_channels * num_heads, hidden_channels)
        self.lin2 = pyg.nn.Linear(hidden_channels, num_labels)

    def forward(self, data):
        """
        Returns the logits for the sample nodes
        """

        x, edge_index = data.x, data.edge_index

        x = F.elu(self.projections(x))

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        x = F.elu(self.lin1(x))
        # x = F.dropout(x, p=0.2, training=self.training)

        return self.lin2(x)

    def projection_layers_l1_norm(self):
        """
        Returns the l1 norm of the projection layers
        """
        return torch.norm(self.projections.weight, p=1)


class BiGAT_sGAT(torch.nn.Module):
    """
    Bipartite graph, single GAT layer
    """

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

        convs = torch.nn.ModuleDict()
        self_loops = torch.nn.ModuleDict()

        self.projections = {
            omic: pyg.nn.Linear(input_dim, proj_dim, weight_initializer="glorot")
            for omic, input_dim in zip(omic_channels, input_dims)
        }

        self.conv1 = GATv2Conv(
            in_channels=proj_dim,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=False,
        )
        self.self_loops2 = {
            name: pyg.nn.Linear(proj_dim, hidden_channels, weight_initializer="glorot")
            for name in omic_channels
        }

        self.conv2 = GATv2Conv(
            in_channels=proj_dim,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=False,
        )
        self.self_loops2 = {
            name: pyg.nn.Linear(proj_dim, hidden_channels, weight_initializer="glorot")
            for name in omic_channels
        }

        self.conv3 = GATv2Conv(
            in_channels=proj_dim,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=False,
        )
        self.self_loops3 = {
            name: pyg.nn.Linear(proj_dim, hidden_channels, weight_initializer="glorot")
            for name in omic_channels
        }


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
        # self.projections = None

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

        # for omic in self.omic_channels:
        #     x_dict[omic] = F.elu(self.projections[omic](x_dict[omic]))
        #     x_dict[omic] = F.dropout(
        #         x_dict[omic], p=self.dropout, training=self.training
        #     )

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

    def move_to_device(self, device):
        for proj_layer in self.projections.values():
            proj_layer.to(device)
        for self_loop in self.self_loops1.values():
            self_loop.to(device)
        for self_loop in self.self_loops2.values():
            self_loop.to(device)
        # for self_loop in self.self_loops3.values():
        #     self_loop.to(device)


class BipartiteRGAT(torch.nn.Module):
    """
    implement the bipartite graph model using RGAT conv
    """

    def __init__(
        self,
        input_sizes,
        proj_dim,
        num_relations,  # dataset.num_relations
        num_heads,  # array of shape (num_layers,)
        hidden_channels,  # array of shape (num_layers,)
        num_omics,
        num_labels,
        feature_integration_mode,
        vcdn_conv_mode=False,
    ):
        """
        Args:
            input_sizes (list): list of input sizes for each omic view
            proj_dim (int): projection dimension
            proj_dropout (float): dropout rate for the projection layers
            num_layers (int): number of RGAT layers
            num_relations (int): number of relations in the dataset
            num_bases (list): number of bases for each RGAT layer
            num_heads (list): number of heads for each RGAT layer
            dropout (list): dropout rate for each RGAT layer
            hidden_channels (list): number of hidden channels for each RGAT layer
            num_labels (int): number of classes
            feature_integration_mode (str): mode for integrating the omic views
            vcdn_conv_mode (bool): whether to use convolutional layers in VCDN
        """
        super().__init__()

        self.projections = torch.nn.ModuleList()
        self.rgat_convs = torch.nn.ModuleList()
        self.self_loops = torch.nn.ModuleList()

        for i in range(len(input_sizes)):
            self.projections.append(pyg.nn.Linear(input_sizes[i], proj_dim))

        self.rgat_conv1 = pyg.nn.RGATConv(
            in_channels=proj_dim,
            out_channels=hidden_channels[0],
            num_relations=num_relations,
            mod="f-scaled",  # cardinality preservation
            heads=num_heads,
        )

        self.self_loops1 = torch.nn.ModuleList()
        for i in range(len(input_sizes)):
            self.self_loops1.append(
                pyg.nn.Linear(proj_dim, hidden_channels[0], weight_initializer="glorot")
            )

        self.rgat_conv2 = pyg.nn.RGATConv(
            in_channels=hidden_channels[0] * num_heads,
            out_channels=hidden_channels[1],
            num_relations=num_relations,
            mod="f-scaled",  # cardinality preservation
            heads=1,  # num_heads,
        )

        self.self_loops2 = torch.nn.ModuleList()
        for i in range(len(input_sizes)):
            self.self_loops2.append(
                pyg.nn.Linear(
                    hidden_channels[0] * num_heads,
                    hidden_channels[1],
                    weight_initializer="glorot",
                )
            )

        self.rgat_conv3 = pyg.nn.RGATConv(
            in_channels=hidden_channels[0] * num_heads,
            out_channels=hidden_channels[1],
            num_relations=num_relations,
            mod="f-scaled",  # cardinality preservation
            heads=1,  # num_heads,
        )

        self.self_loops3 = torch.nn.ModuleList()
        for i in range(len(input_sizes)):
            self.self_loops2.append(
                pyg.nn.Linear(
                    hidden_channels[0] * num_heads,
                    hidden_channels[1],
                    weight_initializer="glorot",
                )
            )

        self.lin1 = pyg.nn.Linear(hidden_channels[1], hidden_channels[1])
        self.lin2 = pyg.nn.Linear(hidden_channels[1], num_labels)

    def forward(self, data: pyg.data.HeteroData):
        """
        Accepts a HeteroData object

        Args:
            data: HeteroData
            omic_layers: list of indices for omic nodes
        """

        # project all input features to the same dimension
        for omic, projection in zip(data.omics, self.projections):
            x = F.elu(projection(data[omic].x))
            data[omic].x = x

        data_hom = data.to_homogeneous()

        x = data_hom.x
        edge_index = data_hom.edge_index
        edge_type = data_hom.edge_type

        x = self.rgat_conv1(x, edge_index, edge_type)  # + self.self_loops1(x)
        for i in range(len(data.omics)):
            x += self.self_loops1[i](data_hom.x[data_hom.node_type == i])

        x = F.elu(x)
        x = self.rgat_conv2(x, edge_index, edge_type)  # + self.self_loops2(x)
        for i in range(len(data.omics)):
            x += self.self_loops1[i](data_hom.x[data_hom.node_type == i])

        x = F.elu(x)

        # collect the sample nodes
        x = self.lin1(x[data_hom.node_type == 0])
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        return self.lin2(x)

    def feature_importance_projection(self, feature_names):
        """
        Given feature names, compute the importance of each feature based on the projection layer
        """

        feature_importances = []

        with torch.no_grad():
            for projection_layer, features in zip(self.projections, feature_names):
                feature_importances.append(
                    torch.abs(projection_layer.weight).sum(dim=1)
                )

        return feature_importances

        # take each omic and stack them to obtain x = (n_omics, n_samples, n_features)
        # x = torch.zeros(
        #     (len(data.omics), data[data.omics[0]].x.shape[0], data_hom.x.shape[1])
        # )

        # assuming that the first n indices correspond to sample nodes
        # for i in range(len(data.omics)):
        #     x[i] = data_hom.x[data_hom.node_type == i]

        # integrate the omic views to obtain the final prediction
        # return self.integration_module(x)

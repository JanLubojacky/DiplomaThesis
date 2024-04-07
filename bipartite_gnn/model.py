import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GATv2Conv

from bipartite_gnn.feat_integration_models import LinearIntegration


class GAT_2L(torch.nn.Module):
    # """"""
    def __init__(self, input_shape, n_classes, channels, heads, dropout=0.1):
        super().__init__()
        torch.manual_seed(1234567)
        self.projections = None
        self.conv1 = GATv2Conv(
            input_shape, channels, heads, dropout=dropout, add_self_loops=False
        )
        self.conv2 = GATv2Conv(
            channels * heads,
            n_classes,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
        )

    def forward(self, data):
        """
        Returns the logits for the sample nodes
        """
        x, edge_index = data.x, data.edge_index

        # x = F.dropout(x, p=0.0, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index)

        return x


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
    ) -> None:
        super().__init__()

        self.omic_channels = omic_channels

        all_names = omic_channels + feature_names

        self.projections = {
            omic: pyg.nn.Linear(input_dim, proj_dim)
            for omic, input_dim in zip(omic_channels, input_dims)
        }

        # omic_channels has names of the omics
        # input_dims has the dimensions of the omics
        # create a dict with the projections whete key is the omic name and value is the projection layer

        self.conv1 = pyg.nn.HeteroConv(
            {
                relation: pyg.nn.GATv2Conv(
                    proj_dim, hidden_channels[0], heads=heads, concat=False
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
                    hidden_channels[0], hidden_channels[1], heads=heads, concat=False
                )
                for relation in relations
            }
        )
        self.self_loops1 = {
            name: pyg.nn.Linear(
                hidden_channels[0], hidden_channels[1], weight_initializer="glorot"
            )
            for name in all_names
        }

        # self.conv3 = pyg.nn.HeteroConv()
        # self.self_loops3 = {}

        self.integration_module = LinearIntegration(
            n_views=len(omic_channels),
            view_dim=hidden_channels[1],
            n_classes=num_classes,
        )

    def forward(self, x_dict, edge_dict):
        """
        Returns the logits for the sample nodes

        Args:
            data: HeteroData

        Returns:
            x (n_samples, n_classes): torch.Tensor
        """

        for omic in self.omic_channels:
            x_dict[omic] = F.relu(self.projections[omic](x_dict[omic]))

        x1 = self.conv1(x_dict, edge_dict)
        for key in x_dict.keys():
            x1[key] = x1[key] + self.self_loops1[key](x_dict[key])
        x1 = x1.relu()

        x2 = self.conv2(x1, edge_dict)
        for key in x_dict.keys():
            x2[key] = x2[key] + self.self_loops2[key](x1[key])
        x2 = x2.relu()

        x_sample_features = torch.cat([x2[omic] for omic in self.omic_channels], dim=1)

        return self.integration_module(x_sample_features)


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
        self.self_loops1 = pyg.nn.Linear(
            proj_dim, hidden_channels[0], weight_initializer="glorot"
        )
        self.rgat_conv2 = pyg.nn.RGATConv(
            in_channels=hidden_channels[0] * num_heads,
            out_channels=hidden_channels[1],
            num_relations=num_relations,
            mod="f-scaled",  # cardinality preservation
            heads=1,  # num_heads,
        )
        self.self_loops2 = pyg.nn.Linear(
            hidden_channels[0], hidden_channels[1], weight_initializer="glorot"
        )

        self.linear = pyg.nn.Linear(hidden_channels[1], num_labels)
        # self.self_loops2 = pyg.nn.Linear(hidden_channels[i], hidden_channels[i + 1])

        # if feature_integration_mode == "linear":
        #     self.integration_module = LinearIntegration(
        #         n_views=len(input_sizes),
        #         view_dim=hidden_channels[-1],
        #         n_classes=num_labels,
        #     )
        # elif feature_integration_mode == "attention":
        #     self.integration_module = AttentionIntegration(
        #         len(input_sizes),
        #         view_dim=hidden_channels[-1],
        #         n_classes=num_labels,
        #     )
        # elif feature_integration_mode == "vcdn":
        #     self.integration_module = VCDN(
        #         len(input_sizes),
        #         hidden_channels[-1],
        #         num_labels,
        #         convolutional=vcdn_conv_mode,
        #     )
        # else:
        #     raise ValueError(
        #         "Unknown feature integration mode, please choose one of linear, attention, vcdn"
        #     )

    def forward(self, data: pyg.data.HeteroData):
        """
        Accepts a HeteroData object

        Args:
            data: HeteroData
            omic_layers: list of indices for omic nodes
        """

        # project all input features to the same dimension
        for omic, projection in zip(data.omics, self.projections):
            x = F.relu(projection(data[omic].x))
            data[omic].x = x

        data_hom = data.to_homogeneous()

        x = data_hom.x
        edge_index = data_hom.edge_index
        edge_type = data_hom.edge_type

        x = self.rgat_conv1(x, edge_index, edge_type)  # + self.self_loops1(x)
        x = x.relu()
        x = self.rgat_conv2(x, edge_index, edge_type)  # + self.self_loops2(x)
        x = x.relu()

        return self.linear(x[data_hom.node_type == 0])

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

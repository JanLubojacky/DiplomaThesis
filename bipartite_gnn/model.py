import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GATv2Conv

from bipartite_gnn.feat_integration_models import (VCDN, AttentionIntegration,
                                                   LinearIntegration)


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


class BipartiteRGAT(torch.nn.Module):
    """
    implement the bipartite graph model using RGAT conv
    """

    def __init__(
        self,
        input_sizes,
        proj_dim,
        # proj_dropout,
        # num_layers,
        num_relations,  # dataset.num_relations
        num_bases,  # array of shape (num_layers,)
        num_heads,  # array of shape (num_layers,)
        dropout,  # array of shape (num_layers,)
        # attention_mode,
        # qkv_dim,
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
        # self.skip_connections = torch.nn.ModuleList() # <- this is probably better than adding manual self-connections
        self.rgat_convs = torch.nn.ModuleList()
        self.self_loops = torch.nn.ModuleList()

        for i in range(len(input_sizes)):
            self.projections.append(pyg.nn.Linear(input_sizes[i], proj_dim))

        for i in range(len(hidden_channels) - 1):
            self.rgat_convs.append(
                pyg.nn.RGATConv(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i + 1],
                    num_relations=num_relations,
                    num_bases=num_bases,
                    # num_blocks=num_blocks,
                    # attention_mode=attention_mode,
                    # mod="f-scaled",  # cardinality preservation
                    heads=num_heads,
                    # dim=qkv_dim, qkv dimension
                    dropout=dropout,
                    # edge_dim, if there is any
                )
            )
            # self.self_loops.append(
            #     pyg.nn.Linear(hidden_channels[i], hidden_channels[i + 1])
            # )

        if feature_integration_mode == "linear":
            self.integration_module = LinearIntegration(
                n_views=len(input_sizes),
                view_dim=hidden_channels[-1],
                n_classes=num_labels,
            )
        elif feature_integration_mode == "attention":
            self.integration_module = AttentionIntegration(
                len(input_sizes),
                view_dim=hidden_channels[-1],
                n_classes=num_labels,
            )
        elif feature_integration_mode == "vcdn":
            self.integration_module = VCDN(
                len(input_sizes),
                hidden_channels[-1],
                num_labels,
                convolutional=vcdn_conv_mode,
            )
        else:
            raise ValueError(
                "Unknown feature integration mode, please choose one of linear, attention, vcdn"
            )

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

    def forward(self, data: pyg.data.HeteroData):
        """
        Accepts a HeteroData object

        Args:
            data: HeteroData
            omic_layers: list of indices for omic nodes
        """

        # project all input features to the same dimension
        for omic, projection in zip(data.omics, self.projections):
            # print(data[omic].x.shape)
            x = F.relu(projection(data[omic].x))
            # x = F.droupout(x, training=self.training)
            data[omic].x = x

        print("x shape after projection:", x.shape)

        data_hom = data.to_homogeneous()

        # Apply RGAT convolutions
        for rgat_conv in self.rgat_convs:
            data_hom.x = rgat_conv(
                data_hom.x, data_hom.edge_index, data_hom.edge_type
            ).relu()

        # take each omic and stack them to obtain x = (n_omics, n_samples, n_features)
        x = torch.zeros(
            (len(data.omics), data[data.omics[0]].x.shape[0], data_hom.x.shape[1])
        )

        print("integration shape", x.shape)

        # assuming that the first n indices correspond to sample nodes
        for i in range(len(data.omics)):
            x[i] = data_hom.x[data_hom.node_type == i]

        # integrate the omic views to obtain the final prediction
        return self.integration_module(x)

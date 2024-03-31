import torch
import torch.nn.functional as F
import torch_geometric as pyg

from bipartite_gnn.feat_integration_models import (VCDN, AttentionIntegration,
                                                   LinearIntegration)


class BiRGAT(torch.nn.Module):
    """
    implement the bipartite graph model using RGAT conv
    """

    def __init__(
        self,
        input_sizes,
        proj_dim,
        proj_dropout,
        num_layers,
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
        super().__init__()

        self.projections = torch.nn.ModuleList()
        # self.skip_connections = torch.nn.ModuleList() # <- this is probably better than adding manual self-connections
        self.rgat_convs = torch.nn.ModuleList()

        for i in range(len(input_sizes)):
            self.projections.append(pyg.nn.Linear(input_sizes[i], proj_dim))

        for i in range(num_layers):
            self.rgat_convs.append(
                pyg.nn.RGATConv(
                    in_channels=-1,
                    out_channels=hidden_channels[i],
                    num_relations=num_relations,
                    num_bases=num_bases[i],
                    # num_blocks=num_blocks,
                    # attention_mode=attention_mode,
                    mod="f-scaled",  # cardinality preservation
                    heads=num_heads[i],
                    # dim=qkv_dim, qkv dimension
                    dropout=dropout[i],
                    # edge_dim, if there is any
                )
            )

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

    def forward(self, data: pyg.data.HeteroData, omic_layers):
        """
        Accepts a HeteroData object

        Args:
            data: HeteroData
            omic_layers: list of indices for omic nodes
        """

        # project all input features to the same dimension
        for omic, projection in zip(data.omics, self.projections):
            x = F.relu(projection(data[omic].x))
            x = F.droupout(x, training=self.training)
            data[omic].x = x

        data_hom = data.to_homogeneous()

        # Apply RGAT convolutions
        for rgat_conv in self.rgat_convs:
            data_hom.x = rgat_conv(data_hom.x, data_hom.edge_index, data_hom.edge_type)

        # take each omic and stack them to obtain x = (n_omics, n_samples, n_features)
        x = torch.zeros([len(omic_layers), data_hom.x[1], data_hom.x[0]])
        for i, layer in enumerate(omic_layers):
            x[i] = data_hom.x[data_hom.node_type[layer]]

        # integrate the omic views to obtain the final prediction
        return self.integration_module(x)

import torch
from torch_geometric.nn.models import GCN, GAT

# from bipartite_gnn.feat_integration_models import LinearIntegration
from bipartite_gnn.feat_integration_models import VCDN


class MOGONET(torch.nn.Module):
    def __init__(
        self,
        omics,
        in_channels,
        hidden_channels,
        num_classes,
        encoder_type,
        dropout=0.4,
        num_layers=2,
        num_heads=2,
        vcdn_hidden_channels=32,
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for i, omic in enumerate(omics):
            if encoder_type == "gcn":
                self.encoders[omic] = GCN(
                    in_channels=in_channels[i],
                    hidden_channels=hidden_channels[omic],
                    out_channels=num_classes,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            elif encoder_type == "gat":
                self.encoders[omic] = GAT(
                    in_channels=in_channels[i],
                    hidden_channels=hidden_channels[omic],
                    out_channels=num_classes,
                    num_layers=num_layers,
                    dropout=dropout,
                    v2=True,
                )
            else:
                raise ValueError(f"Invalid encoder type: {encoder_type}")

        self.integrator = VCDN(
            n_views=len(omics),
            view_dim=num_classes,
            n_classes=num_classes,
            hidden_dim=vcdn_hidden_channels,
        )

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        for omic in data.x_dict.keys():
            x_dict[omic] = self.encoders[omic](x_dict[omic], edge_index_dict[omic])

        # stack all omics on top of each other
        # x.shape = (n_samples, n_omics, n_features)
        x = torch.stack([x_dict[omic] for omic in data.x_dict.keys()], dim=1)

        # x.shape = (n_samples, n_classes)
        x = self.integrator(x)

        return x

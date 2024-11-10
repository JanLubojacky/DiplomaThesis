import torch
from torch_geometric.nn.models import GCN, GAT

from bipartite_gnn.feat_integration_models import (
    LinearIntegration,
    AttentionIntegrator,
    VCDN,
)


class MOGONET(torch.nn.Module):
    def __init__(
        self,
        omics,
        in_channels,
        hidden_channels,
        integration_dim,
        num_classes,
        encoder_type,
        dropout=0.4,
        num_layers=2,
        num_heads=2,
        vcdn_hidden_channels=32,
        integrator_type="linear",
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for i, omic in enumerate(omics):
            if encoder_type == "gcn":
                self.encoders[omic] = GCN(
                    in_channels=in_channels[i],
                    hidden_channels=hidden_channels[omic],
                    out_channels=integration_dim,
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

        if integrator_type == "linear":
            self.integrator = LinearIntegration(
                n_views=len(omic),
                view_dim=integration_dim,
                n_classes=num_classes,
                hidden_dim=vcdn_hidden_channels,
            )
        elif integrator_type == "attention":
            self.integrator = AttentionIntegrator(
                n_views=len(omics),
                view_dim=integration_dim,
                n_classes=num_classes,
                hidden_dim=vcdn_hidden_channels,
                dropout=0.0,
            )
        elif integrator_type == "vcdn":
            self.integrator = VCDN(
                n_views=len(omics),
                view_dim=integration_dim,
                n_classes=num_classes,
                hidden_dim=vcdn_hidden_channels,
            )
        elif integrator_type is None:
            self.integrator = None
        else:
            raise ValueError(
                "Invalid integrator type, please choose from 'linear', 'attention', 'vcdn'"
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
        if self.integrator:
            x = self.integrator(x)
        else:
            x = x.squeeze()

        return x

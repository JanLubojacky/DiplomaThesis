import torch

# from bipartite_gnn.feat_integration_models import LinearIntegration
from bipartite_gnn.feat_integration_models import VCDN


class GCNEncoder:
    def __init__(self, in_channels, hidden_layer, out_channels): ...
    def forward(self, x, edge_index): ...


class GATEncoder:
    def __init__(self, in_channels, hidden_layer, out_channels): ...
    def forward(self, x, edge_index): ...


class MOGONET(torch.nn.module):
    def __init__(
        self,
        omics,
        in_channels,
        hidden_layer,
        out_channels,
        num_classes,
        encoder_type,
        num_heads=2,
    ):
        super().__init__()

        self.encoders = torch.nn.ModelDict()

        if encoder_type == "gcn":
            for omic in omics:
                self.encoders[omic] = GCNEncoder(in_channels, hidden_layer, num_classes)
        elif encoder_type == "gat":
            for omic in omics:
                self.encoders[omic] = GATEncoder(in_channels, hidden_layer, num_classes)
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")

        self.integrator = VCDN(omics, out_channels, num_classes, num_heads)

    def forward(self, x, edge_index): ...

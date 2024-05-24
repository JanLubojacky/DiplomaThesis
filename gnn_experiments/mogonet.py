import torch
from torch_geometric.nn.models import GCN, GAT

# from bipartite_gnn.feat_integration_models import LinearIntegration
# from bipartite_gnn.feat_integration_models import VCDN
from sklearn.metrics import f1_score

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

        # self.integrator = VCDN(
        #     n_views=len(omics),
        #     view_dim=num_classes,
        #     n_classes=num_classes,
        #     hidden_dim=vcdn_hidden_channels,
        # )

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
        if self.integrator is not None:
            x = self.integrator(x)
        else:
            x = x.squeeze()

        return x

    def get_feature_importances(self, data, feature_names, feature_importances):
        """
        Calculates feature importances using a feature ablation approach.

        Args:
            data (Data): PyTorch Geometric Data object containing the input features and labels.
            feature_names (dict): A dictionary with keys as omics and values as lists of feature names.

        Returns:
            feature_importances (dict): A dictionary with keys as omics and values as tensors of feature importances.
        """
        # feature names is a dict with keys as omics and values as feature names
        # after the model is trained perform ablation feature importance
        with torch.no_grad():
            # get the baseline loss
            base_loss = torch.nn.CrossEntropyLoss()(self.forward(data), data.y)

            base_f1 = f1_score(
                data.y.detach().cpu().numpy(),
                torch.argmax(self.forward(data), dim=1).detach().cpu().numpy(),
                average="weighted",
            )

            # print("base_loss", base_loss)
            # print("base_f1", base_f1)
            # return

            # for each omic
            for omic in data.x_dict.keys():
                # feature_importances[omic] = {}

                # for each feature in each omic
                for i, feature_name in enumerate(feature_names[omic]):
                    temp_feature = data.x_dict[omic][
                        :, i
                    ].clone()  # Create a copy of the feature column

                    # ablate the feature
                    data.x_dict[omic][:, i] = 0

                    # run the model
                    y_hat = self.forward(data)

                    # get the loss
                    loss = torch.nn.CrossEntropyLoss()(y_hat, data.y)
                    # print(loss)

                    f1 = f1_score(
                        data.y.detach().cpu().numpy(),
                        torch.argmax(y_hat, dim=1).detach().cpu().numpy(),
                        average="weighted",
                    )
                    # print("basef1, newf1:", base_f1, f1)

                    # store the increase in loss, higher increase -> higher importance
                    if not feature_importances[omic].get(feature_name):
                        feature_importances[omic][feature_name] = loss - base_loss
                    else:
                        feature_importances[omic][feature_name] += loss - base_loss

                    # reset the feature
                    data.x_dict[omic][:, i] = temp_feature

                    # print(data.x_dict[omic][:, i])

                # sort_idx = torch.argsort(feature_importances[omic], descending=True)
                # feature_names[omic] = [feature_names[omic][i] for i in sort_idx]

                # feature_importances[omic] = feature_importances[omic][sort_idx]

                print(f"omic: {omic}")
                print(f"feature_names: {feature_names[omic]}")
                print(f"feature_importances: {feature_importances[omic]}")

        return feature_importances

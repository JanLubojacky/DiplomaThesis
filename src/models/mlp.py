import torch
import torch.nn.functional as F
import pytorch_lightning as L
from torch_geometric.nn import Linear
import logging

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)



class MLPModel(torch.nn.Module):
    def __init__(self, input_sz, num_classes, proj_dim, hidden_channels, dropout):
        super().__init__()
        torch.manual_seed(12345)

        self.proj = Linear(input_sz, proj_dim)
        self.dropout = dropout
        self.hidden_layer = Linear(
            proj_dim,
            hidden_channels,
            weight_initializer="kaiming_uniform",
        )
        self.classifier = Linear(hidden_channels, num_classes, "kaiming_uniform")

    def forward(self, x):
        x = self.proj(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class MLPLightningModule(L.LightningModule):
    def __init__(self, net, lr, l2_lambda):
        super().__init__()
        self.net = net
        self.lr = lr
        self.l2_lambda = l2_lambda

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)

        loss = F.cross_entropy(y_pred, y)

        # any additonal regularization goes here...

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_lambda)
        return optimizer


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

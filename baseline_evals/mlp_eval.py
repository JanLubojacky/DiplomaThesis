import pytorch_lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn import Linear


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        """
        This function initializes the MyDataset class.

        Args:
            X (np.ndarray): The features (data points).
            y (np.ndarray): The labels (target values).
            transform (torchvision.transforms, optional): A transform to apply to the data. Defaults to None.
        """
        self.X = X
        self.y = y
        self.transform = transform  # Optional transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]

        # Apply transform if provided
        if self.transform:
            X_sample = self.transform(X_sample)  # Convert to PyTorch tensor

        return X_sample, y_sample


class MLP(torch.nn.Module):
    def __init__(self, input_sz, num_classes, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(input_sz, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)  # Corrected this line
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class MLPTrainer(L.LightningModule):  # Fixed the import of LightningModule
    def __init__(self, net):  # Changed 'Net' to 'net'
        super().__init__()
        self.net = net  # Changed 'Net' to 'net'

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        loss = F.cross_entropy(y_pred, y)  # Changed 'torch.nn.functional' to 'F'
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        val_loss = F.cross_entropy(y_pred, y)  # Changed 'torch.nn.functional' to 'F'
        val_f1 = f1_score(
            y, y_pred.argmax(dim=1), average="weighted"
        )  # Fixed the calculation of f1_score
        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        test_f1 = f1_score(
            y, y_pred.argmax(dim=1), average="weighted"
        )  # Fixed the calculation of f1_score
        self.log("test_f1", test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler_config": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

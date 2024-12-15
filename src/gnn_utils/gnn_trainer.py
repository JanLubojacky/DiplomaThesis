import torch
import torch_geometric as pyg
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional, Dict, Any
import numpy as np
import os


class GNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        params: Dict[str, Any] = {},
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_model_path=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.save_model_path = save_model_path

        # Initialize best metrics tracking
        self.best_val_score = -float("inf")
        self.best_model_state = None

        # Device selection logic with MPS support
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def validate_data(self, data: pyg.data.HeteroData) -> None:
        """Validate that the data object has required attributes."""
        required_attrs = ["train_mask", "val_mask", "test_mask", "y"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                raise ValueError(f"Data object missing required attribute: {attr}")

        # Validate mask sizes
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            mask = getattr(data, mask_name)
            if mask.sum() == 0:
                raise ValueError(f"Empty mask found: {mask_name}")

    def one_epoch(self, data: pyg.data.HeteroData) -> tuple[float, torch.Tensor]:
        """Run one training epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data)

        loss = self.loss_fn(
            out[data.train_mask],
            data.y[data.train_mask],
        )

        if self.params.get("l1_lambda"):
            l1_loss = self.model.projection_layers_l1_norm()
            loss += l1_loss * self.params["l1_lambda"]

        loss.backward()
        self.optimizer.step()

        return loss.item(), out.detach()

    def test(
        self, data: pyg.data.HeteroData, mask: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Evaluate model on given mask."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            pred = out.argmax(dim=1)[mask].cpu()
            return pred

    def compute_geometric_mean(self, metrics: tuple[float, float, float]) -> float:
        """Compute geometric mean of metrics (accuracy, f1_macro, f1_weighted)."""
        # Add small epsilon to avoid zero values
        epsilon = 1e-10
        metrics_array = np.array(metrics) + epsilon
        return float(np.exp(np.mean(np.log(metrics_array))))

    def save_model(self, save_path: str) -> None:
        """Save the best model state."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            self.model.cpu().state_dict(),
            save_path,
        )
        print(f"Saved new best model to {save_path}")

    def train(
        self,
        data: pyg.data.HeteroData,
        epochs: int,
        log_interval: int = 1,
    ) -> None:
        """Train the model."""
        # Initial validation
        self.validate_data(data)

        # Move data to device once
        data = data.to(self.device)

        for epoch in range(1, epochs + 1):
            # Training step
            train_loss, out = self.one_epoch(data)

            with torch.no_grad():
                # Get validation predictions and compute metrics every epoch
                val_pred = self.test(data, data.val_mask)
                val_metrics = self._compute_metrics(
                    val_pred, data.y[data.val_mask].cpu()
                )

                # Compute geometric mean of validation metrics
                val_score = self.compute_geometric_mean(val_metrics)

                # Save model if validation score improves
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_pred = val_pred
                    print(f"New best validation score: {val_score:.4f}")
                    print(self.save_model_path)
                    if self.save_model_path is not None:
                        self.save_model(self.save_model_path)

                if epoch % log_interval == 0:
                    # Get predictions for all splits
                    train_pred = out.argmax(dim=1)[data.train_mask].cpu()
                    test_pred = self.test(data, data.test_mask)

                    # Compute and log metrics
                    train_metrics = self._compute_metrics(
                        train_pred, data.y[data.train_mask].cpu()
                    )
                    test_metrics = self._compute_metrics(
                        test_pred, data.y[data.test_mask].cpu()
                    )

                    self._log_metrics(
                        epoch, train_loss, train_metrics, val_metrics, test_metrics
                    )

    def _compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[float, float, float]:
        """Compute accuracy and F1 scores."""
        try:
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            return acc, f1_macro, f1_weighted
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            return 0.0, 0.0, 0.0

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: tuple[float, float, float],
        val_metrics: tuple[float, float, float],
        test_metrics: tuple[float, float, float],
    ) -> None:
        """Log training, validation and test metrics."""
        train_acc, train_f1_m, train_f1_w = train_metrics
        val_acc, val_f1_m, val_f1_w = val_metrics
        test_acc, test_f1_m, test_f1_w = test_metrics

        # Calculate geometric means
        val_geom_mean = self.compute_geometric_mean(val_metrics)

        print(f"\nEpoch: {epoch:03d}:")
        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Train F1 Macro: {train_f1_m:.4f}, "
            f"Train F1 Weighted: {train_f1_w:.4f}"
        )
        print(
            f"Val Acc: {val_acc:.4f}, "
            f"Val F1 Macro: {val_f1_m:.4f}, "
            f"Val F1 Weighted: {val_f1_w:.4f}, "
            f"Val Geometric Mean: {val_geom_mean:.4f}"
        )
        print(
            f"Test Acc: {test_acc:.4f}, "
            f"Test F1 Macro: {test_f1_m:.4f}, "
            f"Test F1 Weighted: {test_f1_w:.4f}"
        )
        print("#" * 50)

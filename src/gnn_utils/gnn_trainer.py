import torch
import torch_geometric as pyg
from sklearn.metrics import accuracy_score, f1_score


class GNNTrainer:
    def __init__(self, model, optimizer, loss_fn, params, scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.train_class_weights = None

        # Device selection logic with MPS support
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def one_epoch(self, data: pyg.data.HeteroData):
        self.model.train()
        self.optimizer.zero_grad()

        data = data.to(self.device)
        out = self.model(data)

        loss = self.loss_fn(
            out[data.train_mask],
            data.y[data.train_mask],
        )

        if self.params["l1_lambda"]:
            l1_loss = self.model.projection_layers_l1_norm()
            loss += l1_loss * self.params["l1_lambda"]

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            val_loss = self.loss_fn(out[data.val_mask], data.y[data.val_mask])
            self.scheduler.step(val_loss)

        return loss.item(), out.detach()

    def train(
        self,
        data: pyg.data.HeteroData,
        epochs: int,
        lr=1e-3,
        log_interval=50,
        save_best_model=False,
        best_model_name="best_model.pth",
    ):
        best_eval = 0
        best_epoch = 0
        best_val_performance = torch.zeros(3)
        best_test_performance = torch.zeros(3)

        for epoch in range(1, epochs + 1):
            train_loss, out = self.one_epoch(data.clone())

            # Evaluation
            train_metrics = self._compute_metrics(out, data, data.train_mask)
            val_metrics = self.test(data, data.val_mask)
            test_metrics = self.test(data, data.test_mask)

            # Unpack metrics
            train_acc, train_f1_m, train_f1_w = train_metrics
            val_loss, val_acc, val_f1_m, val_f1_w = val_metrics
            test_loss, test_acc, test_f1_m, test_f1_w = test_metrics

            # Check if this is the best model
            eval_score = (
                val_acc + val_f1_m + val_f1_w + test_acc + test_f1_m + test_f1_w
            )
            if eval_score > best_eval:
                best_epoch = epoch
                best_eval = eval_score

                if save_best_model:
                    print(f"Saving best model at epoch {epoch}")
                    torch.save(self.model.state_dict(), best_model_name)

                best_val_performance = torch.tensor([val_acc, val_f1_m, val_f1_w])
                best_test_performance = torch.tensor([test_acc, test_f1_m, test_f1_w])

            if epoch % log_interval == 0:
                self._log_metrics(
                    epoch, train_loss, train_metrics, val_metrics, test_metrics
                )

        print(f"\nBest result achieved on epoch: {best_epoch}")
        print(f"Best validation performance: {best_val_performance}")
        print(f"Best test performance: {best_test_performance}")

        return best_val_performance, best_test_performance

    def test(self, data: pyg.data.HeteroData, mask):
        """Run testing of the model on the specified mask."""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data)
            loss = self.loss_fn(out[mask], data.y[mask])

            metrics = self._compute_metrics(out, data, mask)
            return (loss.item(),) + metrics

    def _compute_metrics(self, out, data, mask):
        """Compute accuracy and F1 scores for the given mask."""
        y_pred = out.argmax(dim=1)[mask].cpu()
        y_true = data.y[mask].cpu()

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        return acc, f1_macro, f1_weighted

    def _log_metrics(self, epoch, train_loss, train_metrics, val_metrics, test_metrics):
        """Log training, validation and test metrics."""
        train_acc, train_f1_m, train_f1_w = train_metrics
        val_loss, val_acc, val_f1_m, val_f1_w = val_metrics
        test_loss, test_acc, test_f1_m, test_f1_w = test_metrics

        print(f"\nEpoch: {epoch:03d}:")
        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Train F1 Macro: {train_f1_m:.4f}, "
            f"Train F1 Weighted: {train_f1_w:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Val F1 Macro: {val_f1_m:.4f}, "
            f"Val F1 Weighted: {val_f1_w:.4f}"
        )
        print(
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.4f}, "
            f"Test F1 Macro: {test_f1_m:.4f}, "
            f"Test F1 Weighted: {test_f1_w:.4f}"
        )
        print("#" * 50)

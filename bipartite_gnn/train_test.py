import torch
import torch_geometric as pyg
from sklearn.metrics import accuracy_score, f1_score


class GNNTrainer:
    def __init__(self, model, optimizer, params, loss_fn, scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params

    def train(self, data: pyg.data.HeteroData, omic_layers):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data, omic_layers)  # data, omic_layers

        # calculate the loss for the training nodes only
        loss = self.loss_fn(out, data.y[data.train_mask])

        # L1 regularization for the projection layers
        if self.model.projections is not None:
            for proj_layer in self.model.projections:
                l1_reg = torch.tensor(0.0)
                for param in self.model.projections.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += l1_reg * self.params["l1_lambdas"]

        loss.backward()
        self.optimizer.step()
        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def test(self, data: pyg.data.HeteroData, omic_layers, mode: str = "val"):
        """
        Run testing of the BiGNN model

        Args:
            data (pyg.data.HeteroData): HeteroData object
            omic_layers (list): list of indices for omic nodes
            mode (str, optional): "val" or "test". Defaults to "val".
        Returns:
            tuple: loss, accuracy, f1_macro, f1_weighted if mode is "val"
            tuple: accuracy, f1_macro, f1_weighted if mode is "test"
        """

        with torch.no_grad():
            self.model.eval()
            # mask for the validation or test nodes
            mask = data.val_mask if mode == "val" else data.test_mask
            y_pred = self.model(data, omic_layers)
            y_pred = y_pred[mask].to("cpu")
            y_true = data.y[mask]

            # calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")
            f1_weighted = f1_score(y_true, y_pred, average="weighted")

            if mode == "val":
                loss = self.loss_fn(y_pred, y_true)
                return loss, acc, f1_macro, f1_weighted

            return acc, f1_macro, f1_weighted

    def test_feature_importance(
        self, data: pyg.data.HeteroData, omic_layers, mode: str = "reset"
    ):
        """
        Compute feature importance

        given the current test split, shuffle / reset feature values for certain features
        and disconnect the corresponding feature node in the graph, run testing with this
        degenerated model and compute the difference in performance

        Args:
            data (pyg.data.HeteroData): HeteroData object
            omic_layers (list): list of indices for omic nodes
        Returns:
            list: list of tensors with feature importances
        """

        feature_importances = []

        for omic in omic_layers:
            current_feature_importances = torch.zeros(data[omic].x.shape[1])

            for feature in range(data[omic].x.shape[1]):
                # save the original feature values
                original_values = data.x[:, feature].clone()

                # shuffle the feature values
                data.x[:, feature] = torch.zero_like(original_values)

                metrics = self.test(data, omic_layers, mode="test")
                current_feature_importances[feature] = metrics.sum()

            feature_importances.append(current_feature_importances)

        return feature_importances

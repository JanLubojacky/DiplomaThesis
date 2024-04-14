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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def one_epoch(self, data: pyg.data.HeteroData):
        self.model.train()
        self.optimizer.zero_grad()

        self.model.to(self.device)
        self.model.move_to_device(self.device)
        data = data.to(self.device)

        out = self.model(data)

        # calculate the loss for the training nodes only
        #
        # TODO add class weights to the loss function
        loss = self.loss_fn(
            out[data.train_mask],
            data.y[data.train_mask],
        )

        # L1 regularization for the projection layers
        if self.model.projections is not None and self.params["l1_lambda"] > 0.0:
            l1_loss = self.model.projection_layers_l1_norm()
            loss += l1_loss * self.params["l1_lambda"]

        loss.backward()
        self.optimizer.step()

        # scheduler step
        if self.scheduler is not None:
            val_loss = self.loss_fn(out[data.val_mask], data.y[data.val_mask])
            self.scheduler.step(val_loss)

        return loss, out

    def train(self, data: pyg.data.HeteroData, epochs: int, lr=1e-3, log_interval=50):
        """ """

        # Set the device to CUDA if available, otherwise use CPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model and data to the device
        # self.model = self.model.to(self.device)
        # data = data.to(device)

        for epoch in range(1, epochs + 1):
            # very important to clone
            train_loss, out = self.one_epoch(data.clone())

            if epoch % log_interval == 0:
                data = data.to("cpu")

                train_acc = accuracy_score(
                    data.y[data.train_mask],
                    out.argmax(dim=1).to("cpu")[data.train_mask],
                )
                train_f1_m = f1_score(
                    data.y[data.train_mask],
                    out.argmax(dim=1)[data.train_mask].to("cpu"),
                    average="macro",
                )
                train_f1_w = f1_score(
                    data.y[data.train_mask],
                    out.argmax(dim=1)[data.train_mask].to("cpu"),
                    average="weighted",
                )

                val_loss, val_acc, val_f1_m, val_f1_w = self.test(data, data.val_mask)
                test_loss, test_acc, test_f1_m, test_f1_w = self.test(
                    data, data.test_mask
                )

                print(f"Epoch: {epoch:03d}, ")
                print(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1 Macro: {train_f1_m:.4f}, Train F1 Weighted: {train_f1_w:.4f}"
                )
                print(
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1 Macro: {val_f1_m:.4f}, Val F1 Weighted: {val_f1_w:.4f}"
                )
                print(
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1 Macro: {test_f1_m:.4f}, Test F1 Weighted: {test_f1_w:.4f}"
                )
                print("#" * 50)

    def test(self, data: pyg.data.HeteroData, mask):
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
            out = self.model(data.clone())
            y_pred = out.argmax(dim=1)
            y_pred = y_pred[mask].to("cpu")
            y_true = data.y[mask]

            # calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")
            f1_weighted = f1_score(y_true, y_pred, average="weighted")

            loss = self.loss_fn(out[mask], y_true)

            return loss, acc, f1_macro, f1_weighted

    def test_feature_importance(
        self, data: pyg.data.HeteroData, omic_layers, mode: str = "zero"
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

                # return the original values
                data.x[:, feature] = original_values

            feature_importances.append(current_feature_importances)

        return feature_importances


# import torch
# import torch_geometric as pyg
# from sklearn.metrics import accuracy_score, f1_score
#
#
# class GNNTrainer:
#     def __init__(self, model, optimizer, loss_fn, params, scheduler=None):
#         self.model = model
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.params = params
#
#     def one_epoch(self, data: pyg.data.HeteroData):
#         self.model.train()
#         self.optimizer.zero_grad()
#
#         out = self.model(data)
#
#         # calculate the loss for the training nodes only
#         # TODO add class weights to the loss function
#         loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
#
#         # L1 regularization for the projection layers
#         if self.model.projections is not None and self.params["l1_lambda"] > 0.0:
#             l1_reg = torch.tensor(0.0)
#             for proj_layer in self.model.projections:
#                 l1_reg += torch.norm(proj_layer.weight, 1)
#             loss += l1_reg * self.params["l1_lambda"]
#
#         loss.backward()
#         self.optimizer.step()
#
#         # scheduler step
#         if self.scheduler is not None:
#             self.scheduler.step()
#
#         return loss, out
#
#     def train(self, data: pyg.data.HeteroData, epochs: int, lr=1e-3, log_interval=50):
#         """ """
#
#         for epoch in range(1, epochs + 1):
#             # very important to clone
#             train_loss, out = self.one_epoch(data.clone())
#
#             if epoch % log_interval == 0:
#                 train_acc = accuracy_score(
#                     data.y[data.train_mask],
#                     out.argmax(dim=1)[data.train_mask],
#                 )
#                 train_f1_m = f1_score(
#                     data.y[data.train_mask],
#                     out.argmax(dim=1)[data.train_mask],
#                     average="macro",
#                 )
#                 train_f1_w = f1_score(
#                     data.y[data.train_mask],
#                     out.argmax(dim=1)[data.train_mask],
#                     average="weighted",
#                 )
#
#                 val_loss, val_acc, val_f1_m, val_f1_w = self.test(
#                     data.clone(), data.val_mask
#                 )
#                 test_loss, test_acc, test_f1_m, test_f1_w = self.test(
#                     data.clone(), data.test_mask
#                 )
#
#                 print(f"Epoch: {epoch:03d}, ")
#                 print(
#                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1 Macro: {train_f1_m:.4f}, Train F1 Weighted: {train_f1_w:.4f}"
#                 )
#                 print(
#                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1 Macro: {val_f1_m:.4f}, Val F1 Weighted: {val_f1_w:.4f}"
#                 )
#                 print(
#                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1 Macro: {test_f1_m:.4f}, Test F1 Weighted: {test_f1_w:.4f}"
#                 )
#                 print("#" * 50)
#
#     def test(self, data: pyg.data.HeteroData, mask):
#         """
#         Run testing of the BiGNN model
#
#         Args:
#             data (pyg.data.HeteroData): HeteroData object
#             omic_layers (list): list of indices for omic nodes
#             mode (str, optional): "val" or "test". Defaults to "val".
#         Returns:
#             tuple: loss, accuracy, f1_macro, f1_weighted if mode is "val"
#             tuple: accuracy, f1_macro, f1_weighted if mode is "test"
#         """
#
#         with torch.no_grad():
#             self.model.eval()
#             # mask for the validation or test nodes
#             out = self.model(data)
#             y_pred = out.argmax(dim=1)
#             y_pred = y_pred[mask].to("cpu")
#             y_true = data.y[mask]
#
#             # calculate metrics
#             acc = accuracy_score(y_true, y_pred)
#             f1_macro = f1_score(y_true, y_pred, average="macro")
#             f1_weighted = f1_score(y_true, y_pred, average="weighted")
#
#             loss = self.loss_fn(out[mask], y_true)
#
#             return loss, acc, f1_macro, f1_weighted
#
#     def test_feature_importance(
#         self, data: pyg.data.HeteroData, omic_layers, mode: str = "zero"
#     ):
#         """
#         Compute feature importance
#
#         given the current test split, shuffle / reset feature values for certain features
#         and disconnect the corresponding feature node in the graph, run testing with this
#         degenerated model and compute the difference in performance
#
#         Args:
#             data (pyg.data.HeteroData): HeteroData object
#             omic_layers (list): list of indices for omic nodes
#         Returns:
#             list: list of tensors with feature importances
#         """
#
#         feature_importances = []
#
#         for omic in omic_layers:
#             current_feature_importances = torch.zeros(data[omic].x.shape[1])
#
#             for feature in range(data[omic].x.shape[1]):
#                 # save the original feature values
#                 original_values = data.x[:, feature].clone()
#
#                 # shuffle the feature values
#                 data.x[:, feature] = torch.zero_like(original_values)
#
#                 metrics = self.test(data, omic_layers, mode="test")
#                 current_feature_importances[feature] = metrics.sum()
#
#                 # return the original values
#                 data.x[:, feature] = original_values
#
#             feature_importances.append(current_feature_importances)
#
#         return feature_importances

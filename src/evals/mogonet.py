import os
import optuna
import torch
from tqdm import tqdm
from copy import deepcopy

from sklearn.metrics import f1_score, accuracy_score

from src.models.mogonet import MOGONET
from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager
from src.gnn_utils.gnn_trainer import GNNTrainer


class MOGONETEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
        params: dict = {},
        save_model_path: bool = None,
    ):
        """ """
        super().__init__(data_manager, n_trials, verbose)
        self.params = params
        self.model = None
        self.save_model_path = save_model_path

        data, _, _, _ = data_manager.get_split(0)
        self.n_classes = data.y.unique().shape[0]
        self.in_channels = [data.x_dict[omics].shape[1] for omics in data.x_dict.keys()]
        self.omic_names = list(data.x_dict.keys())

    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""

        # we select the hyperparameters manually for now but
        # could also select them from the trial later
        self.model = MOGONET(
            omics=self.omic_names,
            in_channels=self.in_channels,
            num_classes=self.n_classes,
            hidden_channels=self.params["encoder_hidden_channels"],
            encoder_type=self.params["encoder_type"],
            dropout=self.params["dropout"],
            integrator_type=self.params["integrator_type"],
            integration_in_dim=self.params["integration_in_dim"],
            vcdn_hidden_channels=self.params["vcdn_hidden_channels"],
        )
        if self.save_model_path:
            save_model_path = f"{self.save_model_path}_{self.fold_idx}.pt"
        else:
            save_model_path = None
        self.trainer = GNNTrainer(
            model=self.model,
            optimizer=torch.optim.Adam(
                self.model.parameters(), lr=1e-3
            ),  # this could be later set by the trial
            loss_fn=torch.nn.CrossEntropyLoss(),
            # params={
            #     "l1_lambda": 0.01, # this could be later set by the trial
            # }
            save_model_path=save_model_path,
        )

    def train_model(self, data, _) -> None:
        """Train model implementation"""
        self.trainer.train(data, self.params["epochs"], self.params["log_interval"])

    def test_model(self, data, _) -> dict:
        """Test model implementation"""
        # y_pred = self.trainer.test(data, data.test_mask).numpy()
        y_pred = self.trainer.best_pred
        y_true = data.y[data.test_mask].numpy()

        return self._calculate_metrics(y_true, y_pred)

    def load_model(self, load_model_path):
        """Load the model with weights from a checkpoint"""
        self.model.load_state_dict(torch.load(load_model_path, weights_only=True))
        self.model.eval()

    def feature_importance(
        self, model_states_dir: str, n_permutations: int = 5
    ) -> dict:
        """
        Calculate feature importance scores using permutation importance method averaged over multiple shuffles.
        Computes importance across all samples, not just test set.

        Args:
            model_states_dir (str): Directory containing model checkpoint files
            n_permutations (int): Number of random permutations to average over for each feature

        Returns:
            dict: Nested dictionary containing importance scores for each feature in each omic
        """
        feature_importance = {}
        permutation_scores = {}  # Store scores for each permutation

        # Get model checkpoint files
        model_checkpoints = os.listdir(model_states_dir)
        model_checkpoints.sort()

        # Iterate through folds
        for fold_idx in tqdm(range(self.data_manager.n_splits)):
            # Train and test data is together and differentiated by masks for GNNs
            train_x, _, _, _ = self.data_manager.get_split(fold_idx)

            # Load model state for this fold
            self.load_model(f"{model_states_dir}/{model_checkpoints[fold_idx]}")

            # Print scores for this fold
            pred_y = self.model(train_x).argmax(dim=1)
            acc = accuracy_score(train_x.y, pred_y)
            f1_macro = f1_score(train_x.y, pred_y, average="macro")
            f1_weighted = f1_score(train_x.y, pred_y, average="weighted")
            print(
                f"Fold {fold_idx}: Acc: {acc:.4f}, F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}"
            )

            with torch.no_grad():
                # Get baseline predictions and loss for all samples
                baseline_pred = self.model(train_x)
                baseline_loss = self.trainer.loss_fn(baseline_pred, train_x.y).item()

                # For each omic type
                for omic_name in self.omic_names:
                    # Get original feature matrix
                    original_features = train_x.x_dict[omic_name].clone()
                    n_features = original_features.shape[1]
                    feature_names = self.data_manager.feature_names[omic_name]

                    # For each feature in this omic
                    for feature_idx in range(n_features):
                        feature_name = feature_names[feature_idx]
                        if feature_name not in permutation_scores:
                            permutation_scores[feature_name] = []

                        # Perform multiple permutations for this feature
                        for _ in range(n_permutations):
                            perturbed_x = deepcopy(train_x)
                            permuted_feature = original_features[:, feature_idx][
                                torch.randperm(original_features.shape[0])
                            ]
                            perturbed_x.x_dict[omic_name][:, feature_idx] = (
                                permuted_feature
                            )

                            # Get new predictions with permuted feature for all samples
                            perturbed_pred = self.model(perturbed_x)
                            perturbed_loss = self.trainer.loss_fn(
                                perturbed_pred, train_x.y
                            ).item()

                            # Calculate importance score (increase in loss)
                            importance_score = perturbed_loss - baseline_loss
                            permutation_scores[feature_name].append(importance_score)

        # Average the scores across all permutations and folds
        for feature_name, scores in permutation_scores.items():
            feature_importance[feature_name] = sum(scores) / len(scores)

        return feature_importance

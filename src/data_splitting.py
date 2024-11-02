from pathlib import Path

import numpy as np
import polars as pl
from mrmr import mrmr_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class OmicDataSplitter:
    """
    A class to handle cross-validation splits for omic data with preprocessing capabilities.
    Designed for omic data where features (genes) are rows and samples are columns.

    Parameters:
    -----------
    annotations_df : polars.DataFrame
        DataFrame containing sample annotations with 'sample_ids' and 'class' columns
    n_splits : int, default=5
        Number of cross-validation splits
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(self, annotations_df: pl.DataFrame, n_splits: int = 5, random_state: int = 3):
        if not {"sample_ids", "class"}.issubset(annotations_df.columns):
            raise ValueError("annotations_df must contain 'sample_ids' and 'class' columns")

        self.annotations_df = annotations_df
        self.n_splits = n_splits
        self.random_state = random_state

        # Create stratified k-fold splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.splits = list(self.skf.split(annotations_df["sample_ids"], annotations_df["class"]))

    def _verify_sample_columns(self, data_df: pl.DataFrame) -> list:
        """
        Verify sample columns and return their names in the correct order.
        Sample columns are those that match sample_ids in annotations_df.
        """
        sample_ids = set(self.annotations_df["sample_ids"])
        sample_cols = [col for col in data_df.columns if col in sample_ids]

        if len(sample_cols) != len(sample_ids):
            raise ValueError("Not all sample IDs from annotations found in data columns")

        # Reorder sample columns to match annotations order
        ordered_sample_cols = [
            col for col in self.annotations_df["sample_ids"] if col in sample_cols
        ]
        return ordered_sample_cols

    def _get_scaler(self, method: str) -> object:
        """Get the appropriate scaler based on method."""
        if method == "standardization":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError("method must be either 'standardization' or 'minmax'")

    def _normalize_split(self, train_data: np.ndarray, test_data: np.ndarray, method: str) -> tuple:
        """
        Normalize training and test data using the specified method.
        Scaler is fitted only on training data.

        Parameters:
        -----------
        train_data : np.ndarray
            Training data of shape (n_features, n_train_samples)
        test_data : np.ndarray
            Test data of shape (n_features, n_test_samples)
        method : str
            Normalization method ('standardization' or 'minmax')

        Returns:
        --------
        tuple
            (normalized_train_data, normalized_test_data)
        """
        scaler = self._get_scaler(method)

        # Transpose to (n_samples, n_features), fit on train data
        train_normalized = scaler.fit_transform(train_data.T)
        # Transform test data using the same scaler
        test_normalized = scaler.transform(test_data.T)

        # Transpose back to (n_features, n_samples)
        return train_normalized.T, test_normalized.T

    def _select_features_split(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_y: np.ndarray,
        feature_ids: list,
        method: str,
        n_features: int = 100,
    ) -> tuple:
        """
        Apply feature selection using only training data.

        Parameters:
        -----------
        train_data : np.ndarray
            Training data of shape (n_features, n_train_samples)
        test_data : np.ndarray
            Test data of shape (n_features, n_test_samples)
        train_y : np.ndarray
            Training target values
        feature_ids : list
            List of feature identifiers
        method : str
            Feature selection method ('variance' or 'mrmr')
        n_features : int
            Number of features to select when using mrmr

        Returns:
        --------
        tuple
            (selected_train_data, selected_test_data, selected_feature_ids)
        """
        if method == "variance":
            selector = VarianceThreshold()
            # Transpose to (n_samples, n_features), fit on train data
            train_selected = selector.fit_transform(train_data.T)
            # Transform test data using the same selector
            test_selected = selector.transform(test_data.T)
            selected_features = np.array(feature_ids)[selector.get_support()]

            # Transpose back to (n_features, n_samples)
            return train_selected.T, test_selected.T, selected_features

        elif method == "mrmr":
            # Convert training data to pandas for mrmr
            train_df = pl.DataFrame(train_data.T, columns=feature_ids).to_pandas()
            selected_features = mrmr_classif(X=train_df, y=train_y, K=n_features)

            # Get indices of selected features
            selected_indices = [feature_ids.index(f) for f in selected_features]

            # Select features from both train and test data
            train_selected = train_data[selected_indices]
            test_selected = test_data[selected_indices]

            return train_selected, test_selected, selected_features
        else:
            raise ValueError("method must be either 'variance' or 'mrmr'")

    def process_data(
        self,
        data_df: pl.DataFrame,
        output_dir: str,
        annotation_cols: list = ["GENE_ID", "GENE_NAME"],
        normalization: str = None,
        feature_selection: str = None,
        n_features: int = 100,
    ):
        """
        Process the input data and create cross-validation splits.
        Normalization and feature selection are performed independently for each split.

        Parameters:
        -----------
        data_df : polars.DataFrame
            Input data frame where rows are features (genes) and columns are samples
        output_dir : str
            Directory to save the splits
        annotation_cols : list, default=["GENE_ID", "GENE_NAME"]
            List of column names that contain feature annotations
        normalization : str, optional
            Normalization method ('standardization' or 'minmax')
        feature_selection : str, optional
            Feature selection method ('variance' or 'mrmr')
        n_features : int, default=100
            Number of features to select when using mrmr
        """
        # Verify and get ordered sample columns
        sample_cols = self._verify_sample_columns(data_df)

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract feature annotations and data
        feature_annotations = data_df.select(annotation_cols)
        X = data_df.select(sample_cols).to_numpy()  # Shape: (n_features, n_samples)
        y = self.annotations_df["class"].to_numpy()
        feature_ids = feature_annotations.get_column(annotation_cols[0]).to_list()

        # Process each split independently
        for split_idx, (train_idx, test_idx) in enumerate(self.splits):
            # Split the data
            X_train = X[:, train_idx]
            X_test = X[:, test_idx]
            y_train = y[train_idx]

            # Get sample IDs for this split
            train_samples = [sample_cols[i] for i in train_idx]
            test_samples = [sample_cols[i] for i in test_idx]

            # Apply normalization if specified
            if normalization:
                X_train, X_test = self._normalize_split(X_train, X_test, normalization)

            # Apply feature selection if specified
            if feature_selection:
                X_train, X_test, selected_features = self._select_features_split(
                    X_train, X_test, y_train, feature_ids, feature_selection, n_features
                )
                # Update feature annotations to only include selected features
                current_feature_annotations = feature_annotations.filter(
                    pl.col(annotation_cols[0]).is_in(selected_features)
                )
            else:
                current_feature_annotations = feature_annotations

            # Create training DataFrame
            train_data = current_feature_annotations.clone()
            for i, sample in enumerate(train_samples):
                train_data = train_data.with_columns(pl.Series(sample, X_train[:, i]))

            # Create test DataFrame
            test_data = current_feature_annotations.clone()
            for i, sample in enumerate(test_samples):
                test_data = test_data.with_columns(pl.Series(sample, X_test[:, i]))

            # Save splits
            train_data.write_csv(output_path / f"split_{split_idx}_train.csv")
            test_data.write_csv(output_path / f"split_{split_idx}_test.csv")

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split


class MultiOmicDataProcessor:
    def __init__(self, data_root: str, omic_types: list[str], process_params: dict):
        self.data_root = data_root
        self.omic_types = omic_types
        self.process_params = process_params
        self.processed_data = {}
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Create necessary directories if they don't exist"""
        for path in ["raw", "splits"]:
            full_path = os.path.join(self.data_root, path)
            for omic in self.omic_types:
                omic_path = os.path.join(full_path, omic)
                os.makedirs(omic_path, exist_ok=True)

    def load_data(self) -> dict[str, np.ndarray]:
        """Load raw or processed data for each omic type"""
        data = {}
        for omic in self.omic_types:
            processed_path = os.path.join(self.data_root, "processed", omic, "data.npy")
            if os.path.exists(processed_path):
                data[omic] = np.load(processed_path)
            else:
                raw_path = os.path.join(self.data_root, "raw", omic, "data.npy")
                if not os.path.exists(raw_path):
                    raise FileNotFoundError(f"No data found for {omic} at {raw_path}")
                data[omic] = np.load(raw_path)
                data[omic] = self.preprocess_data(data[omic], omic)
                np.save(processed_path, data[omic])
        return data

    def selection(self, data, type: str) -> np.ndarray:
        """Apply selection to the data, based on the selected type"""
        match type:
            case "variance":
                pass
            case "mrmr":
                pass
            case _:
                raise ValueError(f"Unknown selection type: {type}")

    def preprocess_data(self, data: np.ndarray, omic_type: str) -> np.ndarray:
        """Apply normalization and preprocessing based on omic type"""
        params = self.process_params.get(omic_type, {})
        if params.get("standardize", True):
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        return data


class CVSplitManager:
    def __init__(
        self,
        splits_dir: str,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def create_splits(
        self, y: np.ndarray
    ) -> list[dict[str, np.ndarray]]:
        """Create and save stratified CV splits"""
        os.makedirs(self.splits_dir, exist_ok=True)

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(np.zeros_like(y), y)
        ):
            # Further split train_val into train and validation
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.test_size,
                stratify=y[train_val_idx],
                random_state=self.random_state,
            )

            # write as csv into files
            fold_dir = os.path.join(self.splits_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # save idxs
            np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
            np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)
            np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)

    def get_split(self, fold: int) -> dict[str, np.ndarray]:
        """Load existing split"""
        fold_dir = os.path.join(self.splits_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Split directory not found: {fold_dir}")

        split_indices = {}
        for split_type in ["train_idx", "val_idx", "test_idx"]:
            path = os.path.join(fold_dir, f"{split_type}.npy")
            split_indices[split_type] = np.load(path)

        return split_indices

    def load_splits(self) -> list[dict[str, np.ndarray]]:
        """Load existing splits"""
        splits = []
        for fold in range(self.n_splits):
            fold_dir = os.path.join(self.splits_dir, f"fold_{fold}")
            if not os.path.exists(fold_dir):
                raise FileNotFoundError(f"Split directory not found: {fold_dir}")

            split_dict = {}
            for split_type in ["train_idx", "val_idx", "test_idx"]:
                path = os.path.join(fold_dir, f"{split_type}.npy")
                split_dict[split_type] = np.load(path)
            splits.append(split_dict)

        return splits

import os
import polars as pl
from abc import ABC, abstractmethod


class OmicDataLoader(ABC):
    """
    Manages a single omic channel

    Args:
        data_dir (str):
            directory where the data is stored
            for n_splits = 5 expected structure of the data is
            .
            ├── test
            │   ├── test_0.csv
            │   ├── test_1.csv
            │   ├── test_2.csv
            │   ├── test_3.csv
            │   └── test_4.csv
            └── train
                ├── train_0.csv
                ├── train_1.csv
                ├── train_2.csv
                ├── train_3.csv
                └── train_4.csv
            and the expected columns in each split csv file are
            | feature_1 | feature_2 | ... | sample_ids | class |

        n_splits (int): number of cross-validation splits
    """

    def __init__(
        self,
        data_dir: str,
        n_splits: int = 5,
        sample_column: str = "sample_ids",
        class_column: str = "class",
    ):
        self.data_dir = data_dir
        self.n_splits = n_splits
        self.train_files = None
        self.test_files = None

        self.index_data()

    def index_data(self) -> pl.DataFrame:
        """
        Index data on creation
        """
        self.train_files = os.listdir(os.path.join(self.data_dir, "train"))
        self.test_files = os.listdir(os.path.join(self.data_dir, "test"))

        if len(self.train_files) != self.n_splits:
            raise ValueError("Number of train files is not equal to n_splits")
        if len(self.test_files) != self.n_splits:
            raise ValueError("Number of test files is not equal to n_splits")

    def get_fold(self, fold_idx: int):
        """
        Given an index retrieve the train and test dataframe for that fold
        """
        train_df = pl.read_csv(os.path.join(self.data_dir, "train", self.train_files[fold_idx]))
        test_df = pl.read_csv(os.path.join(self.data_dir, "test", self.test_files[fold_idx]))

        return train_df, test_df


class OmicDataManager(ABC):
    """
    Given a dict of OmicDataLoaders, loads train test splits from them
    and composes them into a usable datastructure that is then expected by the model
    """

    def __init__(self, omic_data_loaders: dict[str, OmicDataLoader], n_splits: int = 5):
        self.omic_data_loaders = omic_data_loaders
        self.n_splits = n_splits

    def load_split(self, fold_idx: int):
        omic_data = {}
        for key, omic_data_loader in self.omic_data_loaders.items():
            train_df, test_df = omic_data_loader.get_fold(fold_idx)
            omic_data[key] = {
                "train_df": train_df,
                "test_df": test_df,
            }

    @abstractmethod
    def get_split(self, fold_idx: int):
        """
        Given an index of a fold compose a data format expected by the model
        """
        ...

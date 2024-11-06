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

        self.load_data()

    @abstractmethod
    def load_data(self) -> pl.DataFrame:
        """
        Load data on creation
        """
        ...

    @abstractmethod
    def get_split(self, fold_idx: int):
        """
        Given an index retrieve the split
        """
        ...


class OmicDataManager(ABC):
    """
    Given a dict of OmicDataLoaders, loads train test splits from them
    and composes them into a usable datastructure that is then expected by the model
    """

    def __init__(self, omic_data_loaders: dict[str, OmicDataLoader], n_splits: int = 5):
        self.omic_data_loaders = omic_data_loaders
        self.n_splits = n_splits

    @abstractmethod
    def get_split(self, fold_idx: int):
        """
        Given an index retrieve the split
        """
        ...

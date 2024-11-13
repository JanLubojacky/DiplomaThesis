import os
from abc import ABC, abstractmethod

import polars as pl


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
        self.sample_column = sample_column
        self.class_column = class_column

        self.index_data()

    def index_data(self) -> pl.DataFrame:
        """
        Index data on creation
        """
        self.train_files = os.listdir(os.path.join(self.data_dir, "train"))
        self.test_files = os.listdir(os.path.join(self.data_dir, "test"))
        self.train_files.sort()
        self.test_files.sort()

        if len(self.train_files) != self.n_splits:
            raise ValueError("Number of train files is not equal to n_splits")
        if len(self.test_files) != self.n_splits:
            raise ValueError("Number of test files is not equal to n_splits")

    def get_fold(self, fold_idx: int):
        """
        Given an index retrieve the train and test dataframe for that fold
        """
        train_df = pl.read_csv(
            os.path.join(self.data_dir, "train", self.train_files[fold_idx])
        )
        test_df = pl.read_csv(
            os.path.join(self.data_dir, "test", self.test_files[fold_idx])
        )

        return train_df, test_df


class OmicDataManager(ABC):
    """
    Given a dict of OmicDataLoaders, loads train test splits from them
    and composes them into a usable datastructure that is then expected by the model
    """

    def __init__(self, omic_data_loaders: dict[str, OmicDataLoader], n_splits: int = 5):
        self.omic_data_loaders = omic_data_loaders
        self.n_splits = n_splits
        self.sample_ids = None
        self.train_y = None
        self.test_y = None

    def load_split(self, fold_idx: int) -> dict[str, dict[str, pl.DataFrame]]:
        omic_data = {}
        for omic, omic_data_loader in self.omic_data_loaders.items():
            train_df, test_df = omic_data_loader.get_fold(fold_idx)
            omic_data[omic] = {
                "train_df": train_df,
                "test_df": test_df,
                "sample_column": omic_data_loader.sample_column,
                "class_column": omic_data_loader.class_column,
            }

        return omic_data

    def load_classes(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        sample_column: str,
        class_column: str,
    ):
        """
        Given a train_df a test_df and the names of the sample and class columns
        populates the train_y and test_y attributes and ensures that they are matching
        """
        # load the first sample_ids found
        if self.sample_ids is None:
            self.sample_ids = train_df[sample_column].to_numpy()
        else:
            sample_ids = train_df[sample_column].to_numpy()
            if not (sample_ids == self.sample_ids).all():
                raise ValueError("Sample ids are not matching")

        if train_df.columns != test_df.columns:
            raise ValueError("Columns of train and test df are not matching!!!")

        if self.train_y is None and self.test_y is None:
            self.train_y = train_df[class_column].to_numpy()
            self.test_y = test_df[class_column].to_numpy()
        else:
            if not (self.train_y == train_df["class"].to_numpy()).all():
                raise ValueError("Train y are not matching")
            if not (self.test_y == test_df["class"].to_numpy()).all():
                raise ValueError("Test y are not matching")

    def reset_attributes(self):
        """
        Clear the train_y and test_y attributes, this has to be called after each fold
        """
        self.sample_ids = None
        self.train_y = None
        self.test_y = None

    @abstractmethod
    def get_split(self, fold_idx: int):
        """
        Given an index of a fold compose a data format expected by the model
        """
        ...

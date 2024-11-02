import os

import numpy as np
import tqdm
from sklearn.model_selection import StratifiedKFold


class OmicDataLoader:
    pass


class OmicDataSplitter:
    """
    Given a polars DataFrame, performs cross-validation splits, normalization, and feature selection and saves the results to and output directory.

    The expected format of the input dataframe is something like:

    | GENE_ID | GENE_NAME | sample1 | sample2 | ... | sampleN |
    | :-----: | :-------: | :-----: | :-----: | ... | :-----: |

    then annotation_cols would be ['GENE_ID', 'GENE_NAME']

    and y would be a polars series of lenght N where N is the number of samples

    Args:
        df: polars DataFrame
        y: polars DataFrame, expected keys are sample_ids and class values
        annotation_cols: list of column names that contain feature annotations
        output_dir: output directory
        n_splits: number of cross-validation splits
        random_state: random seed
    """

    def __init__(
        self,
        df: pl.DataFrame,
        y_df: pl.DataFrame,
        annotation_cols: list,
        output_dir: str,
        n_splits: int = 5,
        random_state: int = 3,
        verbose: bool = True,
    ):
        # create the output directory and the path if it doesnt exist
        os.makedirs(output_dir, exist_ok=True)

        # make sure that the columns are aligned
        sample_ids_x = df.columns.remove(annotation_cols)
        sample_ids_y = y_df["sample_ids"].to_list()

        if not all(sample_ids_x == sample_ids_y):
            raise ValueError("sample_ids_x and sample_ids_y are not aligned")

        self.df = df
        self.X = self.df.drop(self.annotation_cols).to_numpy().T
        self.y_df = y_df
        self.annotation_cols = annotation_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir = output_dir


    def normalization(self, X: np.ndarray, type="minmax"):
        """
        Args:
            X (np.ndarray): of shape (n_samples, n_features)
            type (str): minmax or standardization
        """
        #TODO make this modifications to the dataframe instead
        match type:
            case "minmax":
                return (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
            case "standardization":
                return (X - X.mean(axis=1)) / X.std(axis=1)
            case _:
                raise ValueError("type must be either 'minmax' or 'standardization'")


    def feature_selection(self, X: np.ndarray, y: np.ndarray, type = "variance"):
        #TODO this must be modifications to the dataframe too
        match type:
            case "variance":
                pass
            case "mrmr":
                pass
            case _:
                raise ValueError("type must be either 'variance' or 'mrmr'")

    def process_data(self):
        """
        Process the data and save it to the output directory
        """

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        y = self.y_df["class"].to_numpy()

        for i, train_index, test_index in tqdm(
            enumerate(skf.split(np.zeros(y.shape), y)),
            total=self.n_splits,
            desc="Processing folds",
            unit="fold",
        ):
            fold_iterator.set_description(f"Processing fold {i+1}/{self.n_splits}")

            # train test split
            print(train_index)
            print(test_index)

            # normalization
            X = self.normalization(X)

            # feature selection

            # write to output directory

"""
This is used for splitting an omic dataset into folds
the folds are preprocessed and saved to the output directory

The preprocessing includes:
    - split into train and test
    - normalize
    - select features
"""

import os

import numpy as np
import pandas as pd
import polars as pl
from mrmr import mrmr_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm


class OmicDataSplitter:
    """
    Given a polars DataFrame, performs cross-validation splits, normalization, and feature selection and saves the results to and output directory.

    The expected format of the input dataframe is something like:

    | GENE_ID | GENE_NAME | sample1 | sample2 | ... | sampleN |
    | :-----: | :-------: | :-----: | :-----: | ... | :-----: |

    then annotation_cols would be ['GENE_ID', 'GENE_NAME']

    and y would be a polars dataframe with the following columns:

    | sample_ids | class |
    | :--------: | :---- |

    Args:
        df: polars DataFrame
        y: polars DataFrame, expected keys are sample_ids and class values
        annotation_cols: list of column names that contain feature annotations
        output_dir: output directory
        n_splits: number of cross-validation splits
        n_features: number of features to select during feature selection
        random_state: random seed
    """

    def __init__(
        self,
        df: pl.DataFrame,
        y_df: pl.DataFrame,
        annotation_cols: list,
        output_dir: str,
        n_splits: int = 5,
        n_features: int = 100,
        random_state: int = 3,
        verbose: bool = True,
    ):
        # create the output directory and the path if it doesnt exist
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

        sample_ids_x = df.columns
        sample_ids_y = y_df["sample_ids"].to_list()
        n_rows_before = df.shape[1] - len(annotation_cols)

        # out of the columns in df, only keep the ones in sample_ids_y
        df = df.select(annotation_cols + sample_ids_y)
        n_rows_after = df.shape[1] - len(annotation_cols)
        print(f"Only {n_rows_after} samples out of {n_rows_before} found in y_df")

        # make sure that the columns are aligned
        sample_ids_x = df.columns
        for annotation_col in annotation_cols:
            sample_ids_x.remove(annotation_col)
        if not sample_ids_x == sample_ids_y:
            raise ValueError("sample_ids_x and sample_ids_y are not aligned")

        self.df = df
        self.X = self.df.drop(annotation_cols).to_numpy().T
        self.y_df = y_df
        self.n_features = n_features
        self.annotation_cols = annotation_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir = output_dir
        self.feature_names = df[annotation_cols[0]].to_list()

    def normalization(self, X_train: np.ndarray, X_test: np.ndarray, type="minmax"):
        """
        Args:
            X (np.ndarray): of shape (n_samples, n_features)
            type (str): minmax or standardization
        """
        # TODO make this modifications to the dataframe instead
        match type:
            case "minmax":
                scaler = MinMaxScaler()
            case "standardization":
                scaler = StandardScaler()
            case _:
                raise ValueError("type must be either 'minmax' or 'standardization'")

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test

    def feature_selection(
        self, X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray, type="mrmr"
    ) -> (np.ndarray, list):
        """
        Args:
            df (pl.DataFrame): polars DataFrame with the inputs and the samples
            X_train (np.ndarray): of shape (n_samples, n_features)
            X_test (np.ndarray): of shape (n_samples, n_features)
            y (np.ndarray): of shape (n_samples,)
            type (str): variance or mrmr
        """
        match type:
            case "variance":
                raise NotImplementedError()
            case "mrmr":
                # convert the dataframes to pandas for use with mrmr
                train_df = pd.DataFrame(X_train, columns=self.feature_names)
                class_df = pd.Series(y, name="class")
                selected_features = mrmr_classif(train_df, class_df, K=self.n_features)

                # construct polars dataframes with the selected features
                train_df = pl.DataFrame(train_df).select(selected_features)
                test_df = pl.DataFrame(X_test, schema=self.feature_names).select(
                    selected_features
                )
            case _:
                raise ValueError("type must be either 'variance' or 'mrmr'")

        return train_df, test_df

    def process_data(self):
        """
        Process the data and save it to the output directory
        """

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        y = self.y_df["class"].to_numpy()

        for i, (train_index, test_index) in tqdm(
            enumerate(skf.split(np.zeros(y.shape), y)),
            total=self.n_splits,
            desc="Processing folds",
            unit="fold",
        ):
            # train test split
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # normalization
            X_train, X_test = self.normalization(X_train, X_test)

            # feature selection
            train_df, test_df = self.feature_selection(X_train, X_test, y_train)

            # add the samples names and the classes as columns
            train_df = train_df.with_columns(
                pl.Series("sample_ids", self.y_df["sample_ids"].to_numpy()[train_index])
            )
            train_df = train_df.with_columns(pl.Series("class", y_train))
            test_df = test_df.with_columns(
                pl.Series("sample_ids", self.y_df["sample_ids"].to_numpy()[test_index])
            )
            test_df = test_df.with_columns(pl.Series("class", y_test))

            # write to output directory
            train_df.write_csv(os.path.join(self.output_dir, "train", f"train_{i}.csv"))
            test_df.write_csv(os.path.join(self.output_dir, "test", f"test_{i}.csv"))


class FullOmicDataProcessor:
    """
    Processes the full dataset without splitting it into train and test sets.
    Useful for fitting models to the full dataset after evaluation to obtain feature importances.
    """
    def __init__(
        self,
        df: pl.DataFrame,
        y_df: pl.DataFrame,
        annotation_cols: list,
        output_dir: str,
        n_features: int = 100,
    ):
        os.makedirs(output_dir, exist_ok=True)

        sample_ids_x = df.columns
        sample_ids_y = y_df["sample_ids"].to_list()
        n_rows_before = df.shape[1] - len(annotation_cols)

        df = df.select(annotation_cols + sample_ids_y)
        n_rows_after = df.shape[1] - len(annotation_cols)
        print(f"Only {n_rows_after} samples out of {n_rows_before} found in y_df")

        sample_ids_x = df.columns
        for annotation_col in annotation_cols:
            sample_ids_x.remove(annotation_col)
        if not sample_ids_x == sample_ids_y:
            raise ValueError("sample_ids_x and sample_ids_y are not aligned")

        self.df = df
        self.X = self.df.drop(annotation_cols).to_numpy().T
        self.y_df = y_df
        self.n_features = n_features
        self.annotation_cols = annotation_cols
        self.output_dir = output_dir
        self.feature_names = df[annotation_cols[0]].to_list()

    def normalize(self, X: np.ndarray, type="minmax") -> np.ndarray:
        match type:
            case "minmax":
                scaler = MinMaxScaler()
            case "standardization":
                scaler = StandardScaler()
            case _:
                raise ValueError("type must be either 'minmax' or 'standardization'")

        return scaler.fit_transform(X)

    def select_features(
        self, X: np.ndarray, y: np.ndarray, type="mrmr"
    ) -> pl.DataFrame:
        match type:
            case "variance":
                raise NotImplementedError()
            case "mrmr":
                df = pd.DataFrame(X, columns=self.feature_names)
                class_df = pd.Series(y, name="class")
                selected_features = mrmr_classif(df, class_df, K=self.n_features)
                processed_df = pl.DataFrame(df).select(selected_features)
            case _:
                raise ValueError("type must be either 'variance' or 'mrmr'")

        return processed_df

    def process_data(self):
        y = self.y_df["class"].to_numpy()

        # Normalize data
        X_normalized = self.normalize(self.X)

        # Feature selection
        processed_df = self.select_features(X_normalized, y)

        # Add sample IDs and classes
        processed_df = processed_df.with_columns(
            [pl.Series("sample_ids", self.y_df["sample_ids"]), pl.Series("class", y)]
        )

        # Save to output directory
        processed_df.write_csv(os.path.join(self.output_dir, "processed_data.csv"))

        return processed_df

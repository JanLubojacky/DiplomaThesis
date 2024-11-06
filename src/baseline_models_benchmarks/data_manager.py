import polars as pl
from src.base_classes.omic_data_loader import OmicDataLoader, OmicDataManager


class CatOmicDataManager(OmicDataManager):
    """
    The simplest data manager for knn, svm and xgboost,
    takes all the data for different omics and concats them
    into a single dataframe that is then passed to the model
    """

    def __init__(self, omic_data_loaders: dict[str, OmicDataLoader], n_splits: int = 5):
        super().__init__(omic_data_loaders, n_splits)

    def get_split(self, fold_idx: int):
        """
        Given a fold_idx returns train_x, test_x, train_y, test_y where
        train_x and test_x are concats of all the omics
        """
        omic_data = self.load_split(fold_idx)
        train_x = []
        test_x = []

        for omic in omic_data:
            train_df = omic_data[omic]["train_df"]
            test_df = omic_data[omic]["test_df"]
            self.load_classes(train_df, test_df, omic.sample_column, omic.class_column)

            train_df = train_df.drop(omic.class_column, omic.sample_column)
            test_df = test_df.drop(omic.class_column, omic.sample_column)

            train_x.append(train_df.to_numpy())
            test_x.append(test_df.to_numpy())

        train_x = pl.concat(train_x)
        test_x = pl.concat(test_x)

        return train_x, test_x, self.train_y, self.test_y

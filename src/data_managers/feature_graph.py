# import polars as pl
import torch
# import torch_geometric as pyg

from src.base_classes.omic_data_loader import OmicDataLoader, OmicDataManager
# from src.gnn_utils.graph_building import (
#     cosine_similarity_matrix,
#     dense_to_coo,
#     keep_n_neighbours,
#     threshold_matrix,
# )


class FeatureGraphDataManager(OmicDataManager):
    """
    Graph data manager for creating graphs based on sample similarity
    """

    def __init__(
        self, omic_data_loaders: dict[str, OmicDataLoader], params, n_splits: int = 5
    ):
        super().__init__(omic_data_loaders, n_splits)
        self.params = params
        # save num_classes and input dims
        data, _, _, _ = self.get_split(0)
        self.omics = data.omics
        self.feature_names = data.feature_names
        self.n_classes = len(torch.unique(data.y))

    def get_split(self, fold_idx: int):
        """
        Given a fold_idx returns train_x, test_x, train_y, test_y
        """

        # load data

        # compose the graph

        # for each gene create a node with [mrna, meth, cnv]
        # create additional nodes for mirna with [mirna, 0, 0]

        # additionally do what is done by Bingjun Li and Sheida Nabavi in 2023
        # add a parallel network (concat of all features probably easiest?)
        # possibly integrate with a cnn => should be better
        # we could also aggregate the mirnas in the same way methylations
        # are aggregated and add edges for mirnas to the network

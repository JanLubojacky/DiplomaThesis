from typing import Tuple, Dict, List
import polars as pl
import torch
import torch_geometric as pyg
from torch_geometric.transforms import ToUndirected

from src.base_classes.omic_data_loader import OmicDataManager, OmicDataLoader
from src.gnn_utils.graph_building import dense_to_coo, create_diff_exp_connections_norm
from src.gnn_utils.interactions import (
    gg_interactions,
    get_mirna_gene_interactions,
    pp_interactions,
    ensembl_ids_to_gene_names,
)


class BipartiteGraphDataManager(OmicDataManager):
    """
    Graph data manager for BRCA multi-omic data with gene-gene interactions
    """

    def __init__(
        self,
        omic_data_loaders: dict[str, OmicDataLoader],
        params: dict,
        n_splits: int = 5,
    ):
        """
        Initialize the BRCA data manager

        Args:
            omic_data_loaders: Dictionary of data loaders for each omic
            params: Dictionary containing:
                - diff_exp_threshold: Threshold for differential expression
            n_splits: Number of cross-validation splits
        """
        super().__init__(omic_data_loaders, n_splits)
        self.params = params

    def get_split(
        self, fold_idx: int
    ) -> Tuple[pyg.data.HeteroData, pyg.data.HeteroData, None, None]:
        """
        Create a heterogeneous graph for the given fold using pre-split data

        Args:
            fold_idx: Index of the fold to use

        Returns:
            data: HeteroData object containing the graph
            same data object (for compatibility)
            None, None (for compatibility)
        """
        # Load pre-split data
        omic_data = self.load_split(fold_idx)
        data = pyg.data.HeteroData()
        data.feature_names = []

        omic_features = {}

        for omic in omic_data:
            # Get train and test data
            train_df = omic_data[omic]["train_df"]
            test_df = omic_data[omic]["test_df"]
            sample_column = omic_data[omic]["sample_column"]
            class_column = omic_data[omic]["class_column"]

            # Load classes
            self.load_classes(train_df, test_df, sample_column, class_column)

            # Convert to torch tensors
            train_features = train_df.drop(class_column, sample_column).to_torch(
                dtype=pl.Float32
            )
            test_features = test_df.drop(class_column, sample_column).to_torch(
                dtype=pl.Float32
            )
            X = torch.cat([train_features, test_features], dim=0)

            # Store features and gene names
            omic_features[omic] = train_df.drop(
                class_column, sample_column
            ).columns

            # Create sample nodes
            data[omic].x = X

            # Create feature nodes, shape (n_features, n_features)
            data[omic + "_feature"].x = torch.ones(X.shape[1], X.shape[1], dtype=torch.float32)
            data.feature_names += [omic + "_feature"]

            # Create edges for the bipartite graph (sample -> feature)
            A = create_diff_exp_connections_norm(X, multiplier=self.params["diff_exp_thresholds"][omic])
            data[omic, "diff_exp", omic + "_feature"].edge_index = dense_to_coo(A)

        data = ToUndirected()(data)

        # Build graph structure
        data = self._build_hetero_graph(
            data=data, omic_features=omic_features
        )

        # Add global attributes
        train_y = torch.tensor(self.train_y)
        test_y = torch.tensor(self.test_y)
        data.y = torch.cat([train_y, test_y], dim=0)

        # Create masks
        n_train = len(train_y)
        n_test = len(test_y)
        data.train_mask = torch.cat(
            [
                torch.ones(n_train, dtype=torch.bool),
                torch.zeros(n_test, dtype=torch.bool),
            ]
        )
        data.test_mask = torch.cat(
            [
                torch.zeros(n_train, dtype=torch.bool),
                torch.ones(n_test, dtype=torch.bool),
            ]
        )
        data.val_mask = data.test_mask  # Using test set as validation for now

        # Reset for next fold
        self.reset_attributes()

        return data, data, None, None

    def _build_hetero_graph(
        self,
        data: pyg.data.HeteroData,
        omic_features: Dict[str, List[str]],
    ) -> None:
        """Build heterogeneous graph with all omics and their interactions"""

        if "mrna" in omic_features:
            mrna_gene_names = ensembl_ids_to_gene_names(omic_features["mrna"])
            data["mrna_feature", "interacts", "mrna_feature"].edge_index = dense_to_coo(
                self._get_gene_interactions(mrna_gene_names, mrna_gene_names)
            )

        # if "mirna" in omic_features and "circrna" in omic_features:
        #     ...

        # Add miRNA-gene interactions
        if "mirna" in omic_features and "mrna" in omic_features:
            mirna_gene_names = ensembl_ids_to_gene_names(omic_features["mirna"], map_file="interaction_data/gene_id_to_mirna_name.csv")
            mirna_mrna = get_mirna_gene_interactions(
                mirna_gene_names,mrna_gene_names, mirna_mrna_db="interaction_data/mirna_genes_mrna.csv"
            )
            data["mirna_feature", "regulates", "mrna_feature"].edge_index = dense_to_coo(mirna_mrna)

        # if "mirna" in omic_features and "cna" in omic_features:
        #     mirna_cna = get_mirna_gene_interactions(
        #         omic_features["mirna"], omic_features["cna"]
        #     )
        #     data["mirna", "regulates", "cna"].edge_index = dense_to_coo(mirna_cna)
        #
        # # Add mRNA-CNA interactions
        # if "mrna" in omic_features and "cna" in omic_features:
        #     mc_A = self._get_gene_interactions(omic_features["mrna"], omic_features["cna"])
        #     data["mrna", "interacts", "cna"].edge_index = dense_to_coo(mc_A)

        # Make graph undirected
        # data = T.ToUndirected()(data)

        # Add metadata
        data.omics = list(omic_features.keys())
        data.num_relations = len(data.edge_index_dict.keys())

        return data

    def _get_gene_interactions(
        self, genes1: List[str], genes2: List[str] = None
    ) -> torch.Tensor:
        """Get gene-gene interactions from databases"""

        gg_A = gg_interactions(genes1, genes2)
        pp_A = pp_interactions(genes1, genes2)

        return torch.logical_or(gg_A, pp_A).int()

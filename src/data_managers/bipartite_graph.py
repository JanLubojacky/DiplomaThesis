from typing import Tuple, Dict, List
import polars as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T

from src.base_classes.omic_data_loader import OmicDataManager
from src.gnn_utils.graph_building import dense_to_coo
from bipartite_gnn.preprocessing import (
    gg_interactions,
    get_mirna_gene_interactions,
    pp_interactions,
)


class BRCAGraphDataManager(OmicDataManager):
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

        # Process each omic and get gene names
        omic_features = {}
        omic_genes = {}

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
            omic_features[omic] = X
            omic_genes[omic] = train_df.drop(
                class_column, sample_column
            ).columns.to_list()

            # Create data nodes
            data[omic].x = X

        # Build graph structure
        self._build_hetero_graph(
            data=data, omic_features=omic_features, omic_genes=omic_genes
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
        omic_features: Dict[str, torch.Tensor],
        omic_genes: Dict[str, List[str]],
    ) -> None:
        """Build heterogeneous graph with all omics and their interactions"""
        # Add gene-gene interactions within omics
        for omic in ["mrna", "cna"]:
            if omic in omic_genes:
                features_A = self._get_gene_interactions(omic_genes[omic])
                data[omic, "interacts", omic].edge_index = dense_to_coo(features_A)

        # Add miRNA-gene interactions
        if "mirna" in omic_genes and "mrna" in omic_genes:
            mirna_mrna = get_mirna_gene_interactions(
                omic_genes["mirna"], omic_genes["mrna"]
            )
            data["mirna", "regulates", "mrna"].edge_index = dense_to_coo(mirna_mrna)

        if "mirna" in omic_genes and "cna" in omic_genes:
            mirna_cna = get_mirna_gene_interactions(
                omic_genes["mirna"], omic_genes["cna"]
            )
            data["mirna", "regulates", "cna"].edge_index = dense_to_coo(mirna_cna)

        # Add mRNA-CNA interactions
        if "mrna" in omic_genes and "cna" in omic_genes:
            mc_A = self._get_gene_interactions(omic_genes["mrna"], omic_genes["cna"])
            data["mrna", "interacts", "cna"].edge_index = dense_to_coo(mc_A)

        # Make graph undirected
        data = T.ToUndirected()(data)

        # Add metadata
        data.omics = list(omic_features.keys())
        data.num_relations = len(data.edge_index_dict.keys())

    def _get_gene_interactions(
        self, genes1: List[str], genes2: List[str] = None
    ) -> torch.Tensor:
        """Get gene-gene interactions from databases"""
        if genes2 is None:
            genes2 = genes1

        gg_A = gg_interactions(genes1, genes2)
        pp_A = pp_interactions(genes1, genes2)
        return torch.logical_or(gg_A, pp_A).int()

from typing import Tuple, List
import polars as pl
import torch
import torch_geometric as pyg
from torch_geometric.transforms import ToUndirected

from src.base_classes.omic_data_loader import OmicDataManager, OmicDataLoader
from src.gnn_utils.graph_building import dense_to_coo, create_diff_exp_connections_norm
from src.gnn_utils.interactions import (
    gg_interactions,
    pp_interactions,
    get_mirna_gene_interactions,
    get_mirna_genes_circrna_interactions,
    ensembl_ids_to_gene_names,
    tf_links,
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
        ensembl_feature_names=True,
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
        self.ensembl_feature_names = ensembl_feature_names

        # save num_classes and input dims
        data, _, _, _ = self.get_split(0)
        self.omics = data.omics
        self.feature_names = data.feature_names
        self.relations = list(data.edge_index_dict.keys())
        self.input_dims = {
            omic: data.x_dict[omic].shape[1] for omic in data.x_dict.keys()
        }
        self.n_classes = len(torch.unique(data.y))

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
            omic_features[omic] = train_df.drop(class_column, sample_column).columns

            # Create sample nodes
            data[omic].x = X

            # Create feature nodes, shape (n_features, n_features)
            # data[omic + "_feature"].x = torch.ones(
            #     X.shape[1], X.shape[1], dtype=torch.float32
            # )
            data[omic + "_feature"].x = torch.ones(X.shape[1], 128, dtype=torch.float32)

            data.feature_names += [omic + "_feature"]

            # Create edges for the bipartite graph (sample -> feature)
            A = create_diff_exp_connections_norm(
                X, multiplier=self.params["diff_exp_thresholds"][omic]
            )
            data[omic, "diff_exp", omic + "_feature"].edge_index = dense_to_coo(A)

        data = ToUndirected()(data)

        # Build graph structure
        data = self._build_hetero_graph(data=data, omic_features=omic_features)

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
        omic_features: dict[str, list[str]],
    ) -> None:
        """Build heterogeneous graph with all omics and their interactions"""

        if "mrna" in omic_features:
            if self.ensembl_feature_names:
                mrna_gene_names = ensembl_ids_to_gene_names(omic_features["mrna"])
            else:
                mrna_gene_names = omic_features["mrna"]
            data["mrna_feature", "interacts", "mrna_feature"].edge_index = dense_to_coo(
                self._get_gene_interactions(mrna_gene_names, mrna_gene_names)
            )
            if "meth" in omic_features:
                if self.ensembl_feature_names:
                    meth_gene_names = ensembl_ids_to_gene_names(omic_features["meth"])
                else:
                    meth_gene_names = omic_features["meth"]
                data[
                    "mrna_feature", "interacts", "meth_feature"
                ].edge_index = dense_to_coo(
                    self._get_gene_interactions(meth_gene_names, meth_gene_names)
                )
            if "cnv" in omic_features:
                if self.ensembl_feature_names:
                    cnv_gene_names = ensembl_ids_to_gene_names(omic_features["cnv"])
                else:
                    cnv_gene_names = omic_features["cnv"]
                data[
                    "mrna_feature", "interacts", "cnv_feature"
                ].edge_index = dense_to_coo(
                    self._get_gene_interactions(cnv_gene_names, cnv_gene_names)
                )
            if "meth" in omic_features and "cnv" in omic_features:
                data[
                    "meth_feature", "interacts", "cnv_feature"
                ].edge_index = dense_to_coo(
                    self._get_gene_interactions(meth_gene_names, cnv_gene_names)
                )

            # Add miRNA-gene interactions
            if "mirna" in omic_features:
                # mirna_gene_names = ensembl_ids_to_gene_names(
                #     omic_features["mirna"],
                #     map_file="interaction_data/gene_id_to_mirna_name.csv",
                # )
                mirna_names = omic_features["mirna"]
                mirna_mrna = get_mirna_gene_interactions(
                    mirna_names,
                    mrna_gene_names,
                    # mirna_gene_names,
                    # mrna_gene_names,
                    # mirna_mrna_db="interaction_data/mirna_genes_mrna.csv",
                )
                # print(f"mirna gene interactions {mirna_mrna.sum()}")
                data[
                    "mirna_feature", "regulates", "mrna_feature"
                ].edge_index = dense_to_coo(mirna_mrna)

                # circrna-mirna interactions
                if "circrna" in omic_features:
                    mirna_circrna_matrix = get_mirna_genes_circrna_interactions(
                        ensembl_ids=omic_features["mirna"],
                        circrna_names=omic_features["circrna"],
                        mirna_circrna_interactions="interaction_data/circrna_mirna_interactions_mirbase.csv",
                    )

                    data[
                        "circrna_feature", "regulates", "mirna_feature"
                    ].edge_index = dense_to_coo(mirna_circrna_matrix)

        # Add metadata
        data.omics = list(omic_features.keys())
        data.num_relations = len(data.edge_index_dict.keys())

        return data

    def _get_gene_interactions(
        self, genes1: List[str], genes2: List[str]
    ) -> torch.Tensor:
        """Get gene-gene interactions from databases"""

        gg_A = gg_interactions(genes1, genes2)
        pp_A = pp_interactions(genes1, genes2)
        A = torch.logical_or(gg_A, pp_A).int()
        tf_A = tf_links(genes1, genes2)

        A = torch.logical_or(A, tf_A).int()
        # print(f"num interactions {A.sum()}")
        return A

import numpy as np
import polars as pl
import torch
import torch_geometric as pyg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from baseline_evals.feature_selection import class_variational_selection
from bipartite_gnn.graph_building import (
    dense_to_coo,
    create_diff_exp_connections_norm,
    dense_to_attributes,
)
from gnn_experiments.bipartite_gnn import BiRGAT
from bipartite_gnn.train_test import GNNTrainer
from bipartite_gnn.preprocessing import get_interactions


def bipartite_eval(
    input_omics,
    n_input_features,
    input_omics_interactions,
    y,
    params,
    multipliers,
    n_splits=5,
    mrmr_selection_params=None,
    interaction_self_loops=False,
):
    """
    Args:
        input_omics (dict) :
            a dict with omic_name: omic_data (pl.DataFrame)
            each omic data is expected to be a polars dataframe
            with the first column being the gene names
            and the rest of the columns the samples
        n_input_features (dict) :
            a dict with omic_name: n_features
            the number of features to keep for each omic
        y (np.array) : target labels
        params (dict) :
            parameters for moogonet such as encoder_type, hidden_channels, dropout, etc.
        n_splits (int) : number of splits for cross validation
    """

    print(input_omics.keys())

    # prepare 5 folds for cross validation
    skf = StratifiedKFold(n_splits=n_splits)

    best_val = torch.zeros(n_splits, 3)
    best_test = torch.zeros(n_splits, 3)

    # keep track of the number of folds for saving the best model
    fold = 0

    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        print(f"Fold {fold + 1} / {n_splits}")

        val_idx, test_idx = train_test_split(
            test_idx, test_size=0.5, stratify=y[test_idx]
        )

        # use new fold-specific dicts for feature names and omics
        feature_names = {
            omic: input_omics[omic][:, 0].to_numpy() for omic in input_omics.keys()
        }
        omics = {}

        for omic in input_omics.keys():
            if mrmr_selection_params is not None:
                # load preselected features
                omic_df = pl.read_csv(
                    f'{mrmr_selection_params['mrmr_path']}/{omic}/{omic}_fold_{fold}.csv'
                )
                omics[omic] = omic_df[:, 1:].to_numpy().T
                feature_names[omic] = omic_df[:, 0].to_numpy()
            else:
                # do the selection here
                omics[omic] = input_omics[omic][:, 1:].to_numpy().T

                filter_mask = class_variational_selection(
                    omics[omic][train_idx], y[train_idx], n_input_features[omic]
                )

                feature_names[omic] = feature_names[omic][filter_mask]

                omics[omic] = omics[omic][:, filter_mask]

        # build graph for mogonet
        data = pyg.data.HeteroData()

        # assuming that feature selection works better, than projection
        proj_dim = params["proj_dim"]
        print(f"Projection dimension: {proj_dim}")

        data.omics = list(omics.keys())
        data.feature_names = [omic + "_feature" for omic in omics.keys()]

        for omic in omics.keys():
            # create differential expression connections
            diff_exp_A = create_diff_exp_connections_norm(
                # omics[omic],
                torch.tensor(omics[omic], dtype=torch.float32),
                multipliers[omic],
            )

            # normalize
            mms = StandardScaler()
            mms.fit(omics[omic][train_idx])
            omics[omic] = torch.tensor(mms.transform(omics[omic]), dtype=torch.float32)

            # assign node features
            data[omic].x = omics[omic]
            data[omic + "_feature"].x = torch.ones(omics[omic].shape[1], proj_dim)

            # assign edge features
            data[omic, "diff_exp", f"{omic}_feature"].edge_index = dense_to_coo(
                diff_exp_A
            )
            data[omic, "diff_exp", f"{omic}_feature"].edge_attr = dense_to_attributes(
                diff_exp_A
            )

        # create interaction edges
        for omic_pair in input_omics_interactions.keys():
            if interaction_self_loops and omic_pair[0] == omic_pair[1]:
                A = torch.ones(
                    omics[omic_pair[0]].shape[1], omics[omic_pair[1]].shape[1]
                )
            else:
                A = torch.zeros(
                    omics[omic_pair[0]].shape[1], omics[omic_pair[1]].shape[1]
                )

            for interaction_type in input_omics_interactions[omic_pair].keys():
                search_both_cols = False
                if omic_pair[0] == omic_pair[1]:
                    search_both_cols = True

                a = get_interactions(
                    feature_names[omic_pair[0]],
                    feature_names[omic_pair[1]],
                    input_omics_interactions[omic_pair][interaction_type][
                        "interactant_col_1"
                    ],
                    input_omics_interactions[omic_pair][interaction_type][
                        "interactant_col_2"
                    ],
                    interact_file=input_omics_interactions[omic_pair][interaction_type][
                        "interactions_file"
                    ],
                    search_both_cols=search_both_cols,
                )

                # logical or between all the final interaction matrices
                A = torch.logical_or(A, a)

            data[
                f"{omic_pair[0]}_feature", "interacts", f"{omic_pair[1]}_feature"
            ].edge_index = dense_to_coo(A)

        data.y = torch.tensor(y)

        data.train_mask = torch.tensor(train_idx)
        data.val_mask = torch.tensor(val_idx)
        data.test_mask = torch.tensor(test_idx)

        data = pyg.transforms.ToUndirected()(data)

        data.num_relations = len(data.edge_index_dict.keys())

        print(data)
        # return
        # print(list(data.edge_index_dict.keys()))
        # return

        print("relations", list(data.edge_index_dict.keys()))

        # train the model
        model = BiRGAT(
            omic_channels=data.omics,
            feature_names=data.feature_names,
            relations=list(data.edge_index_dict.keys()),
            input_dims={
                omic: data.x_dict[omic].shape[1] for omic in data.x_dict.keys()
            },
            proj_dim=proj_dim,
            hidden_channels=params["hidden_channels"],
            num_classes=len(torch.unique(data.y)),
            heads=params["heads"],
            dropout=params["dropout"],
            attention_dropout=params["attention_dropout"],
            use_proj_module=params["use_proj_module"],
            integrator_type=params["integrator_type"],
            three_layers=params["three_layers"],
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

        ccounts = torch.unique(data.y[data.train_mask], return_counts=True)[1]
        inv_w = 1.0 / ccounts.float()
        class_weights = inv_w / inv_w.sum()

        trainer = GNNTrainer(
            model=model,
            loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-5
            ),
            params={
                "l1_lambda": params["l1_lambda"],
            },
        )

        bp = trainer.train(
            data=data,
            epochs=params["epochs"],
            log_interval=params["log_interval"],
            save_best_model=params["save_best_model"],
            best_model_name=f"fold_{fold}_" + params["best_model_path"]
            if params["save_best_model"]
            else None,
        )

        best_val[fold] = bp[0]
        best_test[fold] = bp[1]

        print(bp[0])
        print(bp[1])

        fold += 1

    return best_val, best_test

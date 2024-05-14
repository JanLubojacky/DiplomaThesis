import numpy as np
import torch
import torch_geometric as pyg
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from baseline_evals.feature_selection import class_variational_selection
from bipartite_gnn.graph_building import (
    dense_to_coo,
)
from gnn_experiments.bipartite_gnn import BiRGAT
from bipartite_gnn.train_test import GNNTrainer


def bipartite_eval(
    input_omics,
    n_input_features,
    input_omics_interactions,
    y,
    params,
    n_splits=5,
    mrmr_features=None,
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
            # remove gene names and convert to numpy
            omics[omic] = input_omics[omic][:, 1:].to_numpy().T

            filter_mask = class_variational_selection(
                omics[omic][train_idx], y[train_idx], n_input_features[omic]
            )

            feature_names[omic] = feature_names[omic][filter_mask]

            omics[omic] = omics[omic][:, filter_mask]

            # improve feature selection with mrmr
            if mrmr_features is not None:
                raise NotImplementedError("mrmr not implemented")

            mms = MinMaxScaler()
            mms.fit(omics[omic][train_idx])

            omics[omic] = torch.tensor(mms.transform(omics[omic]), dtype=torch.float32)

        # build graph for mogonet
        data = pyg.data.HeteroData()

        proj_dim = ...

        data.omics = omics.keys()
        data.feature_names = [omic + "_feature" for omic in omics.keys()]

        for omic in omics.keys():
            # nodes
            data[omic].x = omics[omic]
            data[omic + "_feature"].x = torch.ones(omics[omic].shape[1], proj_dim)

            # create differential expression connections
            diff_exp_A = ...

            # edges
            data[omic].edge_index = dense_to_coo(diff_exp_A)

        # create interaction edges
        for omic1, omic2 in input_omics_interactions.keys():
            # both are gene features, create gene-gene edges
            if omic1 == "gene" and omic2 == "gene":
                ...
            # gene mirna interactions
            elif omic1 == "gene" and omic2 == "mirna":
                ...
            else:
                raise ValueError(
                    "Invalid interaction, please use one of (gene, gene) or (gene, mirna)"
                )

        data.y = torch.tensor(y)

        data.train_mask = torch.tensor(train_idx)
        data.val_mask = torch.tensor(val_idx)
        data.test_mask = torch.tensor(test_idx)

        # train the model
        model = BiRGAT()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

        ccounts = torch.unique(data.y[data.train_mask], return_counts=True)[1]
        inv_w = 1.0 / ccounts.float()
        class_weights = inv_w / inv_w.sum()

        trainer = GNNTrainer(
            model=model,
            loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5
            ),
            params={
                "l1_lambda": None,
            },
        )

        bp = trainer.train(
            data=data,
            epochs=params["epochs"],
            log_interval=params["log_interval"],
            save_best_model=params["save_best_model"],
            best_model_name=params["best_model_path"]
            if params["save_best_model"]
            else None,
        )

        best_val[fold] = bp[0]
        best_test[fold] = bp[1]

        fold += 1

    return best_val, best_test

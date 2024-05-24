import numpy as np
import polars as pl
import torch
import torch_geometric as pyg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from baseline_evals.feature_selection import class_variational_selection
from bipartite_gnn.graph_building import (
    cosine_similarity_matrix,
    threshold_matrix,
    dense_to_coo,
    keep_n_neighbours,
)
from gnn_experiments.mogonet import MOGONET
from bipartite_gnn.train_test import GNNTrainer


def mogonet_eval(
    input_omics,
    y,
    params,
    n_splits=5,
    n_input_features=None,
    mrmr_selection_params=None,
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
        mrmr_selection_params (dict) :
            dict with omic_name: {"n_features": n_features, "feature_col_name": feature_col_name}
            should be set if you want to use mrmr feature selection
    """

    # print(input_omics.keys())

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
            # feature selection with mrmr
            if mrmr_selection_params is not None:
                # this can be used to genereate the features here, but it takes a long time
                # so it is better to do that once and cache that since the folds stay the same
                # X_df_selected = mrmr_selection(
                #     X_df=input_omics[omic],
                #     y=y,
                #     train_mask=train_idx,
                #     n_features=mrmr_selection_params[omic]["n_features"],
                #     feature_col_name=mrmr_selection_params[omic]["feature_col_name"],
                #     n_preselected_features=2000,
                # )
                # feature_names[omic] = X_df_selected.columns[1:]
                # omics[omic] = X_df_selected[:, 1:].to_numpy().T
                print(
                    "Loading",
                    f'{mrmr_selection_params['mrmr_path']}/{omic}/{omic}_fold_{fold}.csv',
                )

                # load preselected features
                omic_df = pl.read_csv(
                    f'{mrmr_selection_params['mrmr_path']}/{omic}/{omic}_fold_{fold}.csv'
                )
                omics[omic] = omic_df[:, 1:].to_numpy().T
                feature_names[omic] = omic_df[:, 0].to_numpy()

            else:
                # remove gene names and convert to numpy
                omics[omic] = input_omics[omic][:, 1:].to_numpy().T

                filter_mask = class_variational_selection(
                    omics[omic][train_idx], y[train_idx], n_input_features[omic]
                )

                feature_names[omic] = feature_names[omic][filter_mask]

                omics[omic] = omics[omic][:, filter_mask]

            # we can try swapping this to after the cosine similarity matrix is built
            # and also try swapping this with StandardScaler
            mms = StandardScaler()
            mms.fit(omics[omic][train_idx])

            omics[omic] = torch.tensor(mms.transform(omics[omic]), dtype=torch.float32)

        # print(omics.keys())
        # print(feature_names)
        # return
        # build graph for mogonet
        data = pyg.data.HeteroData()

        for omic in omics.keys():
            # print(f"Building graph for {omic}")

            A_cos_sim = cosine_similarity_matrix(omics[omic])

            if params["graph_style"] == "threshold":
                A = threshold_matrix(
                    A_cos_sim,
                    self_connections=params["self_connections"],
                    target_avg_degree=params["avg_degree"],
                )
            elif params["graph_style"] == "knn":
                A = keep_n_neighbours(
                    A_cos_sim,
                    params["knn"],
                    self_connections=params["self_connections"],
                )
            else:
                raise ValueError("Invalid graph style")

            data[omic].x = omics[omic]
            data[omic].edge_index = dense_to_coo(A)

        data.y = torch.tensor(y)

        data.train_mask = torch.tensor(train_idx)
        data.val_mask = torch.tensor(val_idx)
        data.test_mask = torch.tensor(test_idx)

        # train the model
        model = MOGONET(
            omics=data.x_dict.keys(),
            in_channels=[data.x_dict[omics].shape[1] for omics in data.x_dict.keys()],
            hidden_channels=params["encoder_hidden_channels"],
            num_classes=params["num_classes"],
            encoder_type=params["encoder_type"],
            dropout=params["dropout"],
            integrator_type=params["integrator_type"],
            integration_dim=params["integration_dim"],
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
                "l1_lambda": None,
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

        fold += 1

    return best_val, best_test

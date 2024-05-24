import io
import numpy as np
import polars as pl
import requests
import torch
from tqdm import tqdm
import mrmr

from baseline_evals.feature_selection import class_variational_selection


def ids_to_gene_names(ids, kind):
    """
    Using mygene.info api convert ensgs / ensps to gene names

    Examples:
        ensemble.gene: ENSG00000121410
        ensemble.protein: ENSP00000344461
        refseq: NM_001195426
        miRBase: hsa-mir-21
    """

    allowed_kinds = ["ensembl.gene", "ensembl.protein", "refseq", "miRBase"]
    if kind not in allowed_kinds:
        raise ValueError(f"invalid kind {kind}, please use one of {allowed_kinds}")

    url = "http://mygene.info/v3/query"
    headers = {"accept": "*/*", "Content-Type": "application/json"}
    params = {"species": ["human"], "fields": "symbol, alias", "size": 1000}
    payload = {"q": ids, "scopes": [kind]}

    # Send POST request
    response = requests.post(url, headers=headers, params=params, json=payload)

    # Check for successful response
    if response.status_code == 200:
        data = response.json()
        # print(data)

        if kind == "miRBase":
            gene_names = [""] * len(data)
            for i, item in enumerate(data):
                if item.get("notfound"):
                    gene_names[i] = None
                    continue
                if "alias" in item:
                    for alias in item["alias"]:
                        if alias[:3] == "hsa":
                            gene_names[i] = alias
        else:
            gene_names = []

            # without aliases for now
            for item in data:
                try:
                    if item.get("notfound"):
                        gene_names.append([None])
                        continue
                    gene_names.append(item["symbol"])
                except Exception:
                    gene_names.append(None)

        return gene_names
    else:
        raise Exception(f"Error fetching gene names: {response.status_code}")


def get_interactions(
    interactants1,
    interactants2,
    interactant1_colname,
    interactant2_colname,
    interact_file,
    search_both_cols=False,
):
    """
    Given two lists of interactants, return a matrix of interactions between them based on the interactions
    Args:
        interactant1 (list) : list of interactant names
        interactant2 (list) : list of interactant names
        intaractant1_colname (str) : column name for interactant1
        intaractant2_colname (str) : column name for interactant2
        file (str) : path to the database with interactions, it is expected to have
        columns intaractant1_colname and intaractant2_colname where each row is an interaction
        search_both_cols (bool) :
            if True, interactants1 and interactant2 are the same thing (gene names) and the
            interactions csv contains interaction like gene_from_i2, gene_from_i1 but not gene_from_i1, gene_from_i2
            and we want to catch the interaction no matter the column order
    Returns:
        A (torch.Tensor), shape (len(interactant1), len(interactant2)) : matrix of interactions between interactants
    """

    interact_df = pl.read_csv(interact_file)

    if not isinstance(interactants1, list):
        interactants1 = list(interactants1)
    if not isinstance(interactants2, list):
        interactants2 = list(interactants2)

    if search_both_cols:
        interactant_list = interactants1 + interactants2

        interact_df = interact_df.filter(
            pl.col(interactant1_colname).is_in(interactant_list)
            & pl.col(interactant2_colname).is_in(interactant_list)
        )
    else:
        interact_df = interact_df.filter(
            pl.col(interactant1_colname).is_in(interactants1)
            & pl.col(interactant2_colname).is_in(interactants2)
        )

    A = torch.zeros((len(interactants1), len(interactants2)))

    interactant1_idx = interact_df.columns.index(interactant1_colname)
    interactant2_idx = interact_df.columns.index(interactant2_colname)

    for row in interact_df.iter_rows():
        # gene1, gene2
        try:
            A[
                interactants1.index(row[interactant1_idx]),
                interactants2.index(row[interactant2_idx]),
            ] = 1
        # gene2, gene1
        except ValueError:
            A[
                interactants1.index(row[interactant2_idx]),
                interactants2.index(row[interactant1_idx]),
            ] = 1

    if A.sum() == 0:
        raise Warning("WARNING: No interactions found, are all inputs correct?")
    else:
        print(f"Found {A.sum()} interactions")

    return A


def get_mirna_gene_interactions(
    mirna_names, mrna_names, mirna_mrna_db="mirna_mrna_interactions_DB.csv"
):
    """
    Args:
        mirna_names (list) : list of mirna names, in mature form, e.g. hsa-miR-99a-5p
        mrna_names (list) : list of gene names such as BTBD3, ELOVL7, etc.
        mirna_mrna_db (str) : path to the database with mirna-mrna interactions, it is expected to have
        columns mirna and gene where each row is a mirna-mrna interaction
    """

    mirna_mrna_df = pl.read_csv(mirna_mrna_db)

    if not isinstance(mirna_names, list):
        mirna_names = list(mirna_names)
    if not isinstance(mrna_names, list):
        mrna_names = list(mrna_names)

    mirna_mrna_df = mirna_mrna_df.filter(
        pl.col("gene").is_in(mrna_names) & pl.col("mirna").is_in(mirna_names)
    )

    mirna_mrna_A = torch.zeros((len(mirna_names), len(mrna_names)))

    print(mirna_mrna_A.shape)
    mirna_idx = mirna_mrna_df.columns.index("mirna")
    gene_idx = mirna_mrna_df.columns.index("gene")

    for row in mirna_mrna_df.iter_rows():
        mirna_mrna_A[
            mirna_names.index(row[mirna_idx]), mrna_names.index(row[gene_idx])
        ] = 1

    if mirna_mrna_A.sum() == 0:
        print("WARNING: No interactions found, are all inputs correct?")

    return mirna_mrna_A


def pp_interactions(gene_list_1, gene_list_2, db_file="string_db/ppi.csv"):
    """
    Given two lists of gene names, return a matrix of interactions between them based on the interactions
    of the proteins they encode

    Args:
        gene_list_1 (list) : list of gene names
        gene_list_2 (list) : list of gene names
        db_file (str) : path to the database with ppi interactions, it is expected to have
        columns gene1 and gene2 where each row is a gene-gene interaction
    Returns:
        A (torch.Tensor), shape (len(gene_list1), len(gene_list_2)) : matrix of interactions between genes
    """

    if not isinstance(gene_list_1, list):
        gene_list_1 = list(gene_list_1)
    if not isinstance(gene_list_2, list):
        gene_list_2 = list(gene_list_2)

    A = torch.zeros((len(gene_list_1), len(gene_list_2)))

    ppi = pl.read_csv(db_file)

    genes = gene_list_1 + gene_list_2

    ppi = ppi.filter(pl.col("gene1").is_in(genes) & pl.col("gene2").is_in(genes))

    g1_idx = ppi.columns.index("gene1")
    g2_idx = ppi.columns.index("gene2")

    for row in ppi.iter_rows():
        try:
            A[gene_list_1.index(row[g1_idx]), gene_list_2.index(row[g2_idx])] = 1
        except ValueError:
            try:
                A[gene_list_1.index(row[g2_idx]), gene_list_2.index(row[g1_idx])] = 1
            except ValueError:
                pass

    if A.sum() == 0:
        print("WARNING: No interactions found, are all inputs correct?")

    return A


def gg_interactions(gene_list_1, gene_list_2, check_all_aliases=False):
    """
    Given two lists of gene names, return a matrix of interactions between them

    Args:
        gene_list_1 (list) : list of gene names
        gene_list_2 (list) : list of gene names
        check_all_aliases (bool) : if True, check all aliases for each gene in the database
    Returns:
        interactions_A (torch.Tensor), shape (len(gene_list1), len(gene_list_2)) : matrix of interactions between genes
    """

    interactions_A = torch.zeros((len(gene_list_1), len(gene_list_2)))

    interaction_data = pl.read_csv("biogrid_preprocessed_data.csv")

    if not isinstance(gene_list_1, list):
        gene_list_1 = list(gene_list_1)
    if not isinstance(gene_list_2, list):
        gene_list_2 = list(gene_list_2)

    gene_list = gene_list_1 + gene_list_2

    if not check_all_aliases:
        a_idx = interaction_data.columns.index("Official Symbol Interactor A")
        b_idx = interaction_data.columns.index("Official Symbol Interactor B")

        interaction_data = interaction_data.filter(
            pl.col("Official Symbol Interactor A").is_in(gene_list)
            & pl.col("Official Symbol Interactor B").is_in(gene_list)
        )

        for row in interaction_data.iter_rows():
            try:
                interactions_A[
                    gene_list_1.index(row[a_idx]), gene_list_2.index(row[b_idx])
                ] = 1
            except ValueError:
                try:
                    interactions_A[
                        gene_list_1.index(row[b_idx]), gene_list_2.index(row[a_idx])
                    ] = 1
                except ValueError:
                    pass

        return interactions_A

    # iterate over each row in the dataframe
    for row in tqdm(interaction_data.iter_rows()):
        name_a = row[0]
        name_b = row[1]
        alias_a = row[2]
        alias_b = row[3]

        names_a = [name_a]
        names_b = [name_b]

        if alias_a:
            names_a += alias_a.split("|")
        if alias_b:
            names_b += alias_b.split("|")

        for gene_a in names_a:
            if gene_a in gene_list:
                for gene_b in names_b:
                    if gene_b in gene_list:
                        interactions_A[
                            gene_list.index(gene_a), gene_list.index(gene_b)
                        ] = 1

    if interactions_A.sum() == 0:
        print("WARNING: No interactions found, are all inputs correct?")

    return interactions_A


def get_gene_gene_interactions(gene_list):
    """
    Retrieve interactions between genes

    api docs: https://wiki.thebiogrid.org/doku.php/biogridrest
    """
    request_url = "https://webservice.thebiogrid.org/interactions/"

    # These parameters can be modified to match any search criteria following
    # the rules outlined in the Wiki: https://wiki.thebiogrid.org/doku.php/biogridrest
    params = {
        "accesskey": "7bcc32a723cda4c96a00c69790c6c26e",
        "format": "tab3",  # Return results in TAB3 format
        "geneList": "|".join(gene_list),  # Must be | separated
        "searchNames": "true",  # Search against official names
        "searchSynonyms": "true",
        "includeInteractors": "false",  # Set to true to get any interaction involving EITHER gene, set to false to get interactions between genes
        "taxId": 9606,  # human genes only
        "selfInteractionsExcluded": "true",  # exclude self interactions
        "includeHeader": "true",
    }

    r = requests.get(request_url, params=params)
    if r.status_code != 200:
        print(f"Response {r.text}")
        raise ValueError(f"Request to BIOGRID failed with code {r.status_code}")
    interactions = r.text
    i_df = pl.read_csv(io.StringIO(interactions), separator="\t")

    if i_df.shape[0] == 10_000:
        print(
            "10k interactions returned, its possible that the api call didnt return all interactions"
        )

    return i_df


def mrmr_selection(
    X_df, y, train_mask, n_features, n_preselected_features, feature_col_name
):
    """
    Given a polars DataFrame X of shape (n_features, n_samples) and an array y of shape (n_samples, )
    Select the top n_features using mrmr feature selection
    """

    feature_names = X_df[feature_col_name].to_numpy()

    # assuming the first column is the feature names
    X = X_df[:, 1:].to_numpy().T

    if n_preselected_features is not None:
        # select_indices = variance_filtering(X, n_features=n_preselected_features)
        select_indices = class_variational_selection(
            X, y, n_features=n_preselected_features
        )
    else:
        select_indices = np.ones(X.shape[1], dtype=bool)

    # preselect features & training samples
    X_train = X[train_mask][:, select_indices]
    print(X_train.shape)

    # createa a dataframe with the selected features
    train_df = pl.DataFrame(X_train)
    train_df.columns = np.array(feature_names)[select_indices].tolist()
    train_df = train_df.with_columns(target=pl.Series(y[train_mask]))

    selected_features = mrmr.polars.mrmr_classif(
        train_df, target_column="target", K=n_features
    )

    X_df_select = X_df.filter(pl.col(feature_col_name).is_in(selected_features))

    return X_df_select


def get_thresholded_expressions(expressions, labels):
    """
    Given expressions for genes across different classes, determine
    if the expression of the gene is different for each class
    and then threshold it based on how much different
    """
    ...


def ensure_same_ordering():
    """
    omic_channels (dict) :
        given a dict with samples as columns, ensure that all dataframes have the same ordering of columns
        sorted alphabetically
    """
    ...


def build_graph(omic_channels, labels):
    """
    build the input graph for the gnn model

    Accepts:
        omic_channels (dict) :
            omic channels is a dict with polars dataframes
                - only certain omics are supported
                - where each column has gene_names as headers
                - and the ordering of the samples in each dataframe is the same
            oc = {
                "mrna" : pl.dataframe (n_features, n_samples),
                "mirna" : pl.dataframe,
                "cna" : pl.dataframe,
                "meth" : pl.dataframe,
                "pirna" : pl.dataframe,
            }
        labels (np.array) : a list of labels used to build
        sample-feature edges in the graph based on
        the expression values accross different tumor types
    """

    # get

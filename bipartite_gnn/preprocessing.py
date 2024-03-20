import io

import numpy as np
import polars as pl
import requests


class FeatureSelection:
    """
    Accepts:
        data (pl.dataframe) : polars dataframe with shape (n_features, n_samples)
    """

    def __init__(
        self,
        data: pl.DataFrame,
        feature_names,
        n_features,
        var_threshold,
        min_expression_value=None,
    ):
        self.data: pl.DataFrame = data
        self.feature_names = feature_names
        self.n_features = n_features
        self.var_threshold = var_threshold
        if min_expression_value:
            self.min_expression_value = min_expression_value
        else:
            self.min_expression_value = n_features * 10

    def var_exp_filtering(self, data):
        """
        filter out low variation and expression features
        """
        # compute variance of each row

        # delete rows with variance below the threshold

    def variatonal_selection(self):
        """
        order genes by variance and select the ones with the biggest variance
        """

        # TODO !!! this has to be redone, but mby keep the existing code

        # return self.data[
        #     self.data.var()  # calculates variances
        #     .melt()  # create dataframe with column names in "variable" and variances in "value"
        #     # sort by variance and select n_features columns with the largest variance
        #     .sort(pl.col("value"), descending=True)[: self.n_features]["variable"]
        # ]

    def class_variational_selection(self, labels):
        """
        take the genes and compute their average expression for each class
        then select n_features genes with the largest interclass variance

        as per Li and Nabavi https://arxiv.org/pdf/2302.12838
        """
        classes = np.unique(labels)

        # empty matrix genes x classes
        genes = np.zeros((self.feature_names.shape, len(classes)))

        # add labels to dataframe
        self.data.with_columns(label=labels)

        # for each label, compute the class mean for each gene
        for i, c in enumerate(classes):
            class_data = self.data.filter(pl.col("label") == c)
            genes[:, i] = class_data.mean().to_numpy()

        gene_vars = genes.var(axis=0)

        # generate ordering of genes based on interclass variance
        class_variance = np.argsort(gene_vars)

        # and select the top n ones
        selected_genes = class_variance[: self.n_features]

        return selected_genes

    def mogonet_selection(self):
        """
        selects features using ANOVA and PCA as described in https://doi.org/10.1038/s41467-021-23774-w
        """
        ...


def ids_to_gene_names(ids, kind):
    """
    Using mygene.info api convert ensgs / ensps to gene names
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
        print(data)

        if kind == "miRBase":
            gene_names = [""] * len(data)
            for i, item in enumerate(data):
                for alias in item["alias"]:
                    if alias[:3] == "hsa":
                        gene_names[i] = alias
        else:
            gene_names = [item["symbol"] for item in data]

        return gene_names
    else:
        raise Exception(f"Error fetching gene names: {response.status_code}")


def get_protein_protein_interactions(gene_names):
    """
    Retrieve interactions between proteins from string db
    """

    # convert ensg to ensp

    # get protein-protein interactions between ensps

    # map back to ensgs and gene names

    #


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


def get_mirna_gene_interactions(mirna_names, gene_names):
    """
    Retrieve interactions between mirnas and genes
    """
    ...


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

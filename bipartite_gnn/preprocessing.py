import io

import polars as pl
import requests


class FeatureSelection:
    """
    Accepts:
        data (pl.dataframe) : polars dataframe with shape (n_samples, n_features)
    """

    def __init__(self, data, n_features, var_threshold, min_expression_value=None):
        self.data = data
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
        ...

    def variatonal_selection(self):
        """
        order genes by variance and select the ones with the biggest variance
        """
        return self.data[
            self.data.var()  # calculates variances
            .melt()  # create dataframe with column names in "variable" and variances in "value"
            # sort by variance and select n_features columns with the largest variance
            .sort(pl.col("value"), descending=True)[: self.n_features]["variable"]
        ]

    def mogonet_selection(self):
        """
        selects features using ANOVA and PCA as described in https://doi.org/10.1038/s41467-021-23774-w
        """
        ...


def gene_names_to_ensembl_ids(gene_names): ...


def get_protein_protein_interactions(gene_names):
    """
    Retrieve interactions between proteins
    """


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
                "mrna" : pl.dataframe,
                "mirna" : pl.dataframe,
                "cna" : pl.dataframe,
                "meth" : pl.dataframe,
                "pirna" : pl.dataframe,
            }
        labels (np.array) : a list of labels used to build
        sample-feature edges in the graph based on
        the expression values accross different tumor types
    """
    ...

def get_gene_gene_interactions(gene_names): ...


def get_mirna_gene_interactions(mirna_names, gene_names): ...


def get_thresholded_expressions(expressions, labels):
    """
    Given expressions for genes across different classes, determine
    if the expression of the gene is different for each class
    and then threshold it based on how much different
    """
    ...


def build_graph():
    """
    build the input graph for the gnn model
    """
    ...

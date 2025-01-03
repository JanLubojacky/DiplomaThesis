import polars as pl
import torch
from tqdm import tqdm
import requests
import json


def gene_names_to_ensembl_ids(gene_names):
    """
    Convert a list of gene names to Ensembl IDs using the Ensembl REST API.

    Args:
        gene_names: List of gene symbols/names to convert

    Returns:
        Dictionary mapping gene names to their Ensembl IDs

    Raises:
        requests.exceptions.RequestException: If API request fails
    """
    server = "https://rest.ensembl.org"
    ext = "/lookup/symbol/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Initialize results dictionary
    gene_id_map = {}

    # Process in batches of 1000 (API limit)
    batch_size = 1000
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i : i + batch_size]

        # Prepare request payload
        payload = {"symbols": batch}

        try:
            r = requests.post(server + ext, headers=headers, data=json.dumps(payload))
            r.raise_for_status()

            # Process response
            decoded = r.json()
            print(f"Decoded response: {decoded}")

            # Extract Ensembl IDs from response
            for gene_name, gene_data in decoded.items():
                if gene_data and "id" in gene_data:
                    gene_id_map[gene_name] = gene_data["id"]
                else:
                    gene_id_map[gene_name] = None

        except requests.exceptions.RequestException as e:
            print(
                f"Error processing batch starting at index {i}: {str(e)}",
                file=sys.stderr,
            )
            raise

    return gene_id_map


def ensembl_ids_to_gene_names(
    ensembl_ids: list[str], map_file="interaction_data/gene_id_map.csv"
):
    """
    Given a list of ensembls ids convert them to gene names
    """
    gene_id_map = {}

    with open(map_file, "r") as f:
        for line in f:
            line = line.strip().split(",")
            gene_id_map[line[0]] = line[1]

    return [gene_id_map[ensembl_id] for ensembl_id in ensembl_ids]


def mature_mirnas_to_mirna_genes(mature_mirnas: list[str]):
    """
    A converts a mature mirna name like hsa-miR-1228-3p to mirna gene name like MIR1228
    """
    return ["".join(mirna.split("-")[1:3]).upper() for mirna in mature_mirnas]


def get_mirna_genes_circrna_interactions(
    circrna_names,
    ensembl_ids=None,
    mirna_circrna_interactions="interaction_data/circrna_mirna_interactions_mirbase.csv",
    mirna_names=None,
):
    if mirna_names is None and ensembl_ids is not None:
        # convert ensemnl ids to mirna names
        mirna_names = ensembl_ids_to_gene_names(
            ensembl_ids, map_file="interaction_data/gene_id_to_mirna_name.csv"
        )
    if mirna_names is None:
        raise ValueError("mirna_names or ensembl_ids must be provided")

    # load circna-mirna interaction file
    circrna_mirna_interactions_df = pl.read_csv(mirna_circrna_interactions)
    circrna_mirna_interactions_df = circrna_mirna_interactions_df.filter(
        pl.col("mirna_genes").is_in(mirna_names)
        & pl.col("circRNA").is_in(circrna_names)
    )

    # create interaction matrix
    A = torch.zeros((len(circrna_names), len(mirna_names)))

    # loop over all interactions
    for mirna, circrna in zip(
        circrna_mirna_interactions_df["mirna_genes"],
        circrna_mirna_interactions_df["circRNA"],
    ):
        mirna_idx = mirna_names.index(mirna)
        circrna_idx = circrna_names.index(circrna)
        A[circrna_idx, mirna_idx] = 1

    return A


def get_mirna_gene_interactions(
    mirna_names,
    mrna_names,
    mirna_mrna_db="interaction_data/mirna_gene.csv",
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

    mirna_idx = mirna_mrna_df.columns.index("mirna")
    gene_idx = mirna_mrna_df.columns.index("gene")

    for row in mirna_mrna_df.iter_rows():
        mirna_mrna_A[
            mirna_names.index(row[mirna_idx]), mrna_names.index(row[gene_idx])
        ] = 1

    if mirna_mrna_A.sum() == 0:
        print("MIRNA-GENE WARNING: No interactions found, are all inputs correct?")

    return mirna_mrna_A


def pp_interactions(
    gene_list_1, gene_list_2, db_file="interaction_data/string_db_ppi.csv"
):
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


def gg_interactions(
    gene_list_1,
    gene_list_2,
    check_all_aliases=False,
    db_file="interaction_data/biogrid_preprocessed_data.csv",
):
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

    interaction_data = pl.read_csv(db_file)

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


def tf_links(
    gene_a,
    gene_b,
    tflinksdb="interaction_data/TFLink_Homo_sapiens_interactions_SS_simpleFormat_v1.0.tsv",
):
    tflinks = pl.read_csv(tflinksdb, separator="\t", infer_schema_length=10000)
    all_genes = gene_a + gene_b
    tflinks.select(
        pl.col("Name.TF").is_in(all_genes) & pl.col("Name.Target").is_in(all_genes)
    )

    A = torch.zeros((len(gene_a), len(gene_b)))

    tf_idx = tflinks.columns.index("Name.TF")
    target_idx = tflinks.columns.index("Name.Target")

    for row in tflinks.iter_rows():
        try:
            A[gene_a.index(row[tf_idx]), gene_b.index(row[target_idx])] = 1
        except ValueError:
            try:
                A[gene_a.index(row[target_idx]), gene_b.index(row[tf_idx])] = 1
            except ValueError:
                pass

    if A.sum() == 0:
        print("WARNING: No interactions found, are all inputs correct?")

    return A

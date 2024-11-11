import polars as pl


def get_common_elements(*lists: list) -> list:
    """
    Find common elements that appear in all provided lists.

    Parameters:
    -----------
    *lists : List[T]
        Variable number of lists to compare

    Returns:
    --------
    List[T]
        List of elements common to all input lists, maintaining order from first list
    """
    if not lists:
        return []

    # Convert all lists to sets for efficient intersection
    sets = [set(lst) for lst in lists]

    # Find intersection of all sets
    common_set = sets[0].intersection(*sets[1:])

    return list(common_set)


def sort_feature_columns(
    df: pl.DataFrame, sample_order: list, id_cols: list
) -> pl.DataFrame:
    """
    Sort columns in a gene expression DataFrame based on a provided sample order.
    Keeps identifier columns first, then sorts the sample columns.

    Parameters:
    -----------
    df : pl.DataFrame
        Input DataFrame with gene expression data
    sample_order : list
        List of sample names in desired order
    id_cols : list, optional
        List of identifier columns to keep first (default: ['GENE_ID', 'GENE_NAME'])

    Returns:
    --------
    pl.DataFrame
        DataFrame with columns sorted according to sample_order
    """
    # Get current sample columns (excluding ID columns)
    sample_cols = [col for col in df.columns if col not in id_cols]

    # Verify all samples in sample_order exist in DataFrame
    missing_samples = set(sample_order) - set(sample_cols)
    if missing_samples:
        raise ValueError(
            f"Samples in order list not found in DataFrame: {missing_samples}"
        )

    # Verify all samples in DataFrame are in sample_order
    extra_samples = set(sample_cols) - set(sample_order)
    if extra_samples:
        raise ValueError(
            f"Samples in DataFrame not found in order list: {extra_samples}"
        )

    # Verify all id_cols exist in DataFrame
    missing_id_cols = set(id_cols) - set(df.columns)
    if missing_id_cols:
        raise ValueError(f"ID columns not found in DataFrame: {missing_id_cols}")

    # Create final column order
    final_order = id_cols + sample_order

    # Return DataFrame with sorted columns
    return df.select(final_order)

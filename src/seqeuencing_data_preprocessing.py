import numpy as np
import polars as pl


def preprocess_seq_data(
    df: pl.DataFrame,
    annotation_cols: list[str],
    min_median_expression: float = 100,
    min_detection_rate: float = 0.2,  # detected in 20% of samples
    top_n: int = 5000,
    output_path: str = None,
    cpm_normalization: bool = True,
    log_transform: bool = True,
) -> pl.DataFrame:
    """
    Preprocess RNA sequencing data by filtering, normalizing, and transforming expression values.

    Args:
        df: Input DataFrame containing gene expression data
        annotation_cols: List of column names containing annotations (e.g., ['GENE_ID', 'GENE_NAME'])
        min_median_expression: Minimum median expression threshold for filtering genes
        min_detection_rate: Minimum fraction of samples where gene must be detected
        top_n: Number of top expressed genes to keep if more than top_n pass trough the filters
        output_path: Optional path to save the processed data as CSV

    Returns:
        Processed Polars DataFrame with filtered, normalized, and log2-transformed expression values
    """
    # Validate annotation columns exist in dataframe
    missing_cols = [col for col in annotation_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Annotation columns not found in dataframe: {missing_cols}")

    # Get expression data excluding annotation columns
    expr_df = df.select(pl.all().exclude(*annotation_cols))

    # Calculate detection rate (fraction of samples where gene is expressed)
    detection_rate = (expr_df > 0).sum_horizontal() / len(expr_df.columns)

    # Calculate coefficient of variation
    expr_means = expr_df.mean_horizontal()
    expr_vars = expr_df.select((pl.all() - expr_means).pow(2)).sum_horizontal() / len(
        expr_df.columns
    )
    expr_vars = expr_vars.sqrt()
    coeff_of_var = expr_vars / expr_means

    # Add CV and detection rate to original dataframe
    processed_df = df.with_columns(
        [coeff_of_var.alias("cov"), detection_rate.alias("detection_rate")]
    )

    # Filter and sort based on expression, detection rate, and CV
    processed_df = (
        processed_df.with_columns(
            [
                pl.concat_list(
                    pl.all().exclude(*annotation_cols, "cov", "detection_rate")
                )
                .map_elements(lambda x: np.median(x), return_dtype=pl.Float64)
                .alias("median_expression")
            ]
        )
        # Apply both filters: median expression and detection rate
        .filter(
            (pl.col("median_expression") > min_median_expression)
            & (pl.col("detection_rate") > min_detection_rate)
        )
        .sort("cov", descending=True)
        .limit(top_n)
    )

    # Remove temporary columns
    processed_df = processed_df.drop("median_expression", "detection_rate", "cov")

    # Normalize to counts per million
    if cpm_normalization:
        processed_df = processed_df.with_columns(
            [
                (
                    pl.all().exclude(*annotation_cols)
                    * 10**6
                    / pl.all().exclude(*annotation_cols).sum()
                )
            ]
        )

    # Log2 transform
    if log_transform:
        processed_df = processed_df.with_columns(
            [(pl.all().exclude(*annotation_cols) + 1).log(base=2)]
        )

    # Save to file if path is provided
    if output_path:
        processed_df.write_csv(output_path)

    return processed_df

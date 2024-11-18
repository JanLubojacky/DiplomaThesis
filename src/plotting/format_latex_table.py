import polars as pl
import numpy as np


def generate_metrics_table(
    file_path, table_header="Model Performance Metrics", metrics_mapping=None
):
    """
    Generate a LaTeX table from a CSV file containing model metrics.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the metrics
    table_header : str
        Caption for the LaTeX table
    metrics_mapping : dict, optional
        Dictionary mapping metric names to their column names in the CSV.
        If None, uses default mapping for accuracy and F1 scores.

    Returns:
    --------
    str
        LaTeX formatted table string
    """
    # Default metrics mapping if none provided
    if metrics_mapping is None:
        metrics_mapping = {
            "Accuracy": ("Accuracy", "acc_std"),
            "F1 Macro": ("Macro F1", "f1_macro_std"),
            "F1 Weighted": ("Weighted F1", "f1_weighted_std"),
        }

    # Read the CSV file
    df = pl.read_csv(file_path)

    # Function to format a metric with its std deviation
    def format_metric_std(metric, std, is_bold=False):
        formatted = f"{metric:.3f} $\\pm$ {std:.3f}"
        if is_bold:
            return f"\\textbf{{{formatted}}}"
        return formatted

    # Get the metrics and their standard deviations
    metrics_data = {
        metric_name: df.select([col_name, std_name]).iter_rows()
        for metric_name, (col_name, std_name) in metrics_mapping.items()
    }

    # Store all metrics to find the maximum values
    all_metrics = {
        metric_name: df[col_name].to_list()
        for metric_name, (col_name, _) in metrics_mapping.items()
    }

    # Find maximum values for each metric
    max_values = {metric: max(values) for metric, values in all_metrics.items()}

    # Get models list
    models = df["model"].to_list()

    # Generate LaTeX table header
    latex_table = (
        "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|"
        + "c|" * (len(metrics_mapping) + 1)
        + "}\n\\hline\n"
    )

    # Add column headers
    headers = ["Model"] + list(metrics_mapping.keys())
    latex_table += " & ".join(headers) + " \\\\\n\\hline\n"

    # Reset metrics iterators
    metrics_data = {
        metric_name: df.select([col_name, std_name]).iter_rows()
        for metric_name, (col_name, std_name) in metrics_mapping.items()
    }

    # Add rows
    for model in models:
        row = [model]
        for metric_name in metrics_mapping.keys():
            metric, std = next(metrics_data[metric_name])
            # Check if this is the highest value for this metric
            is_max = np.isclose(metric, max_values[metric_name])
            row.append(format_metric_std(metric, std, is_max))
        latex_table += " & ".join(row) + " \\\\\n"

    # Close the table
    latex_table += f"""\\hline
\\end{{tabular}}
\\caption{{{table_header}}}
\\label{{tab:model-metrics}}
\\end{{table}}"""

    return latex_table

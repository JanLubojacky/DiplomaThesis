import matplotlib.pyplot as plt
import seaborn as sns


def plot_top_genes(importance_dict: dict, n_genes=10, title=None, save_file=None):
    """
    Create a horizontal bar plot of the most important genes directly from a dictionary.

    Parameters:
    importance_dict (dict): Dictionary with gene names as keys and importance scores as values
    n_genes (int): Number of top genes to display
    title (str, optional): Title for saving the plot
    """
    # Sort genes by absolute importance and get top n_genes
    sorted_genes = sorted(
        importance_dict.items(), key=lambda x: x[1], reverse=True
    )[:n_genes]

    # Sort by actual importance value for plotting
    sorted_genes = sorted(sorted_genes, key=lambda x: x[1])

    # Separate gene names and importance values
    gene_names = [gene for gene, _ in sorted_genes]
    importance_values = [value for _, value in sorted_genes]

    # Set up the plot style with adjusted figure size
    plt.figure(figsize=(12, n_genes * 0.25))
    sns.set_style("whitegrid")

    # Create the bar plot
    ax = plt.gca()
    bars = ax.barh(
        range(len(gene_names)),
        importance_values,
        color=sns.color_palette("RdBu_r", n_colors=len(gene_names)),
        height=0.8,
    )

    # Set y-axis labels (gene names)
    ax.set_yticks(range(len(gene_names)))
    ax.set_yticklabels(gene_names, fontsize=12)

    # Customize the plot
    plt.title(title, pad=20, size=16)
    plt.xlabel("Feature Importance Score", size=14)
    plt.ylabel("Gene Name", size=14)

    # Add value labels on the bars with adjusted spacing
    max_val = max(abs(min(importance_values)), abs(max(importance_values)))
    offset = max_val * 0.01  # Dynamic offset based on data range

    for i, value in enumerate(importance_values):
        ax.text(
            value + (offset if value >= 0 else -offset),
            i,
            f"{value:.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=12,
        )

    # Adjust layout and margins
    plt.margins(x=0.1)  # Add 20% padding on x-axis
    plt.tight_layout()

    # save plot
    if save_file:
        plt.savefig(save_file, dpi=400)

    return plt.gcf()

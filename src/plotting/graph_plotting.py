import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


def create_multi_omic_network(
    mrna_dict, mirna_dict, circrna_dict, mrna_A, mirna_gene_A, circrna_mirna_A
):
    """
    Create a multi-omic network graph, atm this takes only mRNA, miRNA and circRNA nodes
    and the respective interactions
    """
    # Create graph
    G = nx.Graph()

    # Add nodes with attributes
    # mRNA nodes (rectangle)
    for gene, importance in mrna_dict.items():
        G.add_node(gene, omic_type="mRNA", importance=importance, shape="s")  # square

    # miRNA nodes (circle)
    for mirna, importance in mirna_dict.items():
        G.add_node(mirna, omic_type="miRNA", importance=importance, shape="o")  # circle

    # circRNA nodes (triangle)
    for circrna, importance in circrna_dict.items():
        G.add_node(
            circrna, omic_type="circRNA", importance=importance, shape="^"
        )  # triangle

    # Add edges with different types
    # mRNA-mRNA interactions
    mrna_genes = list(mrna_dict.keys())
    for i in range(len(mrna_genes)):
        for j in range(i + 1, len(mrna_genes)):
            if mrna_A[i, j] == 1:
                G.add_edge(mrna_genes[i], mrna_genes[j], edge_type="mRNA-mRNA")

    # miRNA-mRNA interactions
    mirna_genes = list(mirna_dict.keys())
    for i in range(len(mirna_genes)):
        for j in range(len(mrna_genes)):
            if mirna_gene_A[i, j] == 1:
                G.add_edge(mirna_genes[i], mrna_genes[j], edge_type="miRNA-mRNA")

    # circRNA-miRNA interactions
    circrna_names = list(circrna_dict.keys())
    for i in range(len(circrna_names)):
        for j in range(len(mirna_genes)):
            if circrna_mirna_A[i, j] == 1:
                G.add_edge(circrna_names[i], mirna_genes[j], edge_type="circRNA-miRNA")

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def plot_multi_omic_network(G, figsize=(20, 10), seed=42):
    # Create figure with a specific layout for the colorbar
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 20, figure=fig)
    ax_main = fig.add_subplot(gs[:, :19])  # Main plot takes up most of the space
    ax_cbar = fig.add_subplot(gs[:, 19])  # Colorbar on the right

    # Set up layout
    pos = nx.fruchterman_reingold_layout(G, scale=1, k=5, iterations=1000, seed=seed)

    # Define edge colors
    edge_colors = {"mRNA-mRNA": "red", "miRNA-mRNA": "blue", "circRNA-miRNA": "green"}

    # Draw edges by type
    for edge_type in edge_colors:
        edge_list = [
            (u, v) for (u, v, d) in G.edges(data=True) if d["edge_type"] == edge_type
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_list,
            edge_color=edge_colors[edge_type],
            alpha=0.5,
            ax=ax_main,
        )

    # Get importance values for color mapping
    node_importances = nx.get_node_attributes(G, "importance")
    max_importance = max(node_importances.values())
    min_importance = min(node_importances.values())

    # Create colormap normalization
    norm = Normalize(vmin=min_importance, vmax=max_importance)

    # Draw nodes by omic type
    for omic_type in ["mRNA", "miRNA", "circRNA"]:
        nodes = [
            node for node, attr in G.nodes(data=True) if attr["omic_type"] == omic_type
        ]
        if not nodes:  # Skip if no nodes of this type
            continue

        # Get node sizes and colors based on importance
        node_sizes = [2000 * node_importances[node] / max_importance for node in nodes]
        node_colors = [plt.cm.viridis(norm(node_importances[node])) for node in nodes]

        # Get node shapes
        node_shapes = [G.nodes[node]["shape"] for node in nodes]
        unique_shape = node_shapes[0]  # All nodes of same type have same shape

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_size=node_sizes,
            node_color=node_colors,
            node_shape=unique_shape,
            alpha=0.7,
            ax=ax_main,
        )

    # Add labels with smaller font size for better visibility
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax_main)

    # Add colorbar
    ColorbarBase(ax_cbar, cmap=plt.cm.viridis, norm=norm, label="Feature Importance")

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=edge_colors["mRNA-mRNA"], label="mRNA-mRNA"),
        plt.Line2D([0], [0], color=edge_colors["miRNA-mRNA"], label="miRNA-mRNA"),
        plt.Line2D([0], [0], color=edge_colors["circRNA-miRNA"], label="circRNA-miRNA"),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="mRNA",
            markerfacecolor="gray",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="miRNA",
            markerfacecolor="gray",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="circRNA",
            markerfacecolor="gray",
            markersize=10,
        ),
    ]
    ax_main.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    ax_main.set_title("Multi-omic Interaction Network")
    ax_main.axis("off")

    plt.tight_layout()
    return plt

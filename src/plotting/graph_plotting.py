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
        for j in range(len(mrna_genes)):
            # if i == j:
            #     continue
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

    # Find nodes that only have self-loops
    nodes_to_remove = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        # Check if the node only has itself as a neighbor
        if len(neighbors) == 1 and neighbors[0] == node:
            nodes_to_remove.append(node)

    # Remove the identified nodes
    G.remove_nodes_from(nodes_to_remove)

    return G


def plot_multi_omic_network(
    G,
    max_iter,
    figsize=(20, 20),
    seed=42,
    title="Multi-omic Interaction Network",
    jitter_tolerance=0.5,
    gravity=0.5,
    colorbar_fontsize=16,
):
    # Create figure with a specific layout for the colorbar
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, figsize[0], figure=fig)
    ax_main = fig.add_subplot(
        gs[:, : figsize[0] - 1]
    )  # Main plot takes up most of the space
    ax_cbar = fig.add_subplot(gs[:, figsize[0] - 1])  # Colorbar on the right

    # Set up layout
    pos = nx.forceatlas2_layout(
        G,
        jitter_tolerance=jitter_tolerance,
        gravity=gravity,
        max_iter=max_iter,
        strong_gravity=True,
        dissuade_hubs=False,
        seed=seed,
    )

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
        node_sizes = [5000 * node_importances[node] / max_importance for node in nodes]
        node_colors = [plt.cm.RdYlBu_r(norm(node_importances[node])) for node in nodes]

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
    nx.draw_networkx_labels(G, pos, font_size=16, ax=ax_main)

    # Add colorbar
    cbar = ColorbarBase(ax_cbar, cmap=plt.cm.RdYlBu_r, norm=norm)
    # Modify colorbar label font size
    cbar.ax.set_ylabel("Feature Importance", fontsize=colorbar_fontsize + 2)
    # Modify colorbar tick labels font size
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

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
            markersize=14,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="miRNA",
            markerfacecolor="gray",
            markersize=14,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="circRNA",
            markerfacecolor="gray",
            markersize=14,
        ),
    ]
    ax_main.legend(
        handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1), fontsize=16
    )

    ax_main.set_title(title, fontsize=26)
    ax_main.axis("off")

    plt.tight_layout()
    return plt

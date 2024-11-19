import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_metrics(
    experiment_files: dict,
    metrics=["Accuracy", "Macro F1", "Weighted F1"],
    metrics_std=["acc_std", "f1_macro_std", "f1_weighted_std"],
    save_file: str | None = None,
):
    results = {}
    first_df = pl.read_csv(next(iter(experiment_files.values())))
    models = first_df["model"].to_list()

    # Process data
    for exp_name, file_path in experiment_files.items():
        df = pl.read_csv(file_path)
        results[exp_name] = {}
        for model in models:
            model_data = df.filter(pl.col("model") == model)
            results[exp_name][model] = {
                "means": model_data.select(pl.col(metrics)).mean().to_numpy().flatten(),
                "stds": model_data.select(pl.col(metrics_std))
                .mean()
                .to_numpy()
                .flatten(),
            }

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    fig.subplots_adjust(wspace=0.3)

    exp_names = list(results.keys())
    x = np.arange(len(exp_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    width = 0.12  # Width of bars

    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        # Calculate positions for grouped bars
        offsets = np.linspace(
            -(len(models) - 1) * width / 2, (len(models) - 1) * width / 2, len(models)
        )

        for model_idx, model in enumerate(models):
            metric_values = [results[exp][model]["means"][idx] for exp in exp_names]
            metric_errors = [results[exp][model]["stds"][idx] for exp in exp_names]

            # Create bars with error bars
            ax.bar(
                x + offsets[model_idx],
                metric_values,
                width,
                yerr=metric_errors,
                label=model,
                color=colors[model_idx],
                capsize=4,
                error_kw={"elinewidth": 1},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=0, fontsize=12)
        ax.set_title(metric)
        ax.grid(True, alpha=0.2)

        # Set y-axis limits
        all_values = [
            results[exp][model]["means"][idx] for exp in exp_names for model in models
        ]
        all_errors = [
            results[exp][model]["stds"][idx] for exp in exp_names for model in models
        ]
        ymin = 0.5  # min(all_values) - max(all_errors)# - 0.05
        ymax = 1.0  # max(all_values) + max(all_errors)#  + 0.05
        ax.set_ylim(ymin, ymax)

        # Add legend
        ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=400)

    return fig



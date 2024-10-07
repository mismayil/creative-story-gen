import argparse
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

sns.set_theme(style="whitegrid")

FIG_SIZE = (80, 20)
XLABEL_SIZE = 50
YLABEL_SIZE = 50
TITLE_SIZE = 70
TICK_SIZE = 40
LEGEND_TITLE_SIZE = 40
LEGEND_SIZE = 30

def get_model_id(metrics, model_family="by_name"):
    model_id = "human"

    if "model_id" in metrics.columns:
        model_id = metrics["model_id"].iloc[0].split("_")[0]
    
    if model_family == "by_type":
        pass
    elif model_family == "by_sentience":
        return "human" if model_id == "human" else "machine"

    return model_id

def plot_metrics_by_item_id(metrics_lst, metric, output_dir, output_format="png", model_family="by_name"):
    for metrics in metrics_lst:
        metrics["model"] = get_model_id(metrics, model_family)

    output_dir = pathlib.Path(output_dir) / f"{output_format}_figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_metrics = pd.concat(metrics_lst)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(metric)
    ax.set_xlabel("Item ID", size=XLABEL_SIZE)
    ax.set_ylabel(metric, size=YLABEL_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.title.set_size(TITLE_SIZE)
    sns.barplot(data=merged_metrics, x="item_id", y=metric, hue="model", ax=ax, err_kws={'linewidth': 5})
    ax.legend(title="Model", title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
    plt.tight_layout()

    plot_path = f"{output_dir}/fig_{metric}_by_item_id.{output_format}"
    print(f"Saving figure to {plot_path}")
    plt.savefig(plot_path)

def plot_metrics_n_gram_diversity(metrics_lst, output_dir, output_format="png", model_family="by_name"):
    for metrics in metrics_lst:
        metrics["model"] = get_model_id(metrics, model_family)

    output_dir = pathlib.Path(output_dir) / f"{output_format}_figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_metrics = pd.concat(metrics_lst)
    group_ids = merged_metrics["group_id"].unique()

    for group_id in group_ids:
        group_metrics = merged_metrics[merged_metrics["group_id"] == group_id].copy()
        group_metrics["metric_corpus_n_gram_diversity"] = group_metrics["metric_corpus_n_gram_diversity"].apply(lambda x: ast.literal_eval(x))
        group_metrics["n_gram"] = [list(range(1, len(group_metrics["metric_corpus_n_gram_diversity"].iloc[0])+1)) for _ in range(len(group_metrics))]
        group_metrics = group_metrics.explode(["metric_corpus_n_gram_diversity", "n_gram"])
        group_metrics["n_gram"] = group_metrics["n_gram"].astype(str)

        metric = "metric_corpus_n_gram_diversity"
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.set_title(metric)
        ax.set_xlabel("n-gram size", size=XLABEL_SIZE)
        ax.set_ylabel(metric, size=YLABEL_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.title.set_size(TITLE_SIZE)
        sns.lineplot(data=group_metrics, x="n_gram", y=metric, hue="model_id", style="model_id", markers=True, lw=5, ax=ax)
        ax.legend(title="Model", title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
        plt.tight_layout()

        plot_path = f"{output_dir}/fig_{metric}_{group_id}.{output_format}"
        print(f"Saving figure to {plot_path}")
        plt.savefig(plot_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-files", type=str, nargs="+", help="Path to csv metrics file(s)", required=True)
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory to save plots", default=None)
    parser.add_argument("-f", "--output-format", type=str, choices=["png", "pdf", "svg"], default="png", help="Format to save the plots in.")
    parser.add_argument("-p", "--plot-types", type=str, nargs="+", choices=["by_item_id", "n_gram_diversity"], default=["by_item_id"], help="Type of plots to generate")
    parser.add_argument("-mf", "--model-family", type=str, choices=["by_name", "by_type", "by_sentience"], default="by_name", help="Model family to group by")

    args = parser.parse_args()

    metrics_lst = []

    for metrics_file in args.input_files:
        metrics_path = pathlib.Path(metrics_file)

        try:
            if metrics_path.suffix == ".csv":
                metrics = pd.read_csv(metrics_path)
                metrics_lst.append(metrics)
            else:
                raise ValueError("Input file extension not supported.")
        except FileNotFoundError:
            print(f"Skipping. Input file not found at {metrics_path}")
            return

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = metrics_path.parent / "figures"
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    for plot_type in args.plot_types:
        if plot_type == "by_item_id":
            for col in metrics_lst[0].columns:
                if col.startswith("metric_"):
                    plot_metrics_by_item_id(metrics_lst, col, output_dir, output_format=args.output_format, model_family=args.model_family)
        elif plot_type == "n_gram_diversity":
            plot_metrics_n_gram_diversity(metrics_lst, output_dir, output_format=args.output_format, model_family=args.model_family)

if __name__ == "__main__":
    main()
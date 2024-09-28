import argparse
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

FIG_SIZE = (80, 20)
XLABEL_SIZE = 50
YLABEL_SIZE = 50
TITLE_SIZE = 70
TICK_SIZE = 40
LEGEND_TITLE_SIZE = 40
LEGEND_SIZE = 30

def get_model_id(metrics):
    if "model_id" in metrics.columns:
        return metrics["model_id"].iloc[0].split("_")[0]
    return "human"

def plot_metrics_by_item_id(metrics_lst, metric, output_dir, output_format="png"):
    for metrics in metrics_lst:
        metrics["model"] = get_model_id(metrics)

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
    ax.legend(title="Model", title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plot_path = f"{output_dir}/fig_{metric}_by_item_id.{output_format}"
    print(f"Saving figure to {plot_path}")
    plt.savefig(plot_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-files", type=str, nargs="+", help="Path to csv metrics file(s)", required=True)
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory to save plots", default=None)
    parser.add_argument("-f", "--output-format", type=str, choices=["png", "pdf", "svg"], default="png", help="Format to save the plots in.")
    parser.add_argument("-p", "--plot-types", type=str, nargs="+", choices=["by_item_id"], default=["by_item_id"], help="Type of plots to generate")

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
                    plot_metrics_by_item_id(metrics_lst, col, output_dir, output_format=args.output_format)

if __name__ == "__main__":
    main()
import argparse
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from pandas.api.types import is_numeric_dtype

from utils import concat_dfs, MODEL_SIZE_MAP, get_model_family

sns.set_theme(style="dark")

FIG_SIZE = (25, 20)
XLABEL_SIZE = 40
YLABEL_SIZE = 40
TITLE_SIZE = 50
TICK_SIZE = 40
LEGEND_TITLE_SIZE = 30
LEGEND_SIZE = 33
PALETTE = "BrBG"
X_ROTATION = 0

LABEL_MAP = {
    "id": "Item set",
    "model": "Model",
    "sentience": "Subject",
    "semantic_distance": "Semantic distance",
}

MODEL_NAME_MAP = {
    "gpt-4": "GPT-4",
    "gemini-1.5-flash": "Gemini-1.5",
    "claude-3-5-sonnet-20240620": "Claude-3.5",
    "llama-3.1-405b-instruct": "Llama-3.1-405B",
    "human": "Human",
    "gemini-1.5": "Gemini-1.5",
    "claude-3-5": "Claude-3.5",
    "llama-3.1": "Llama-3.1",
    "machine": "AI"
}

colors = ['#a16518', '#dbb972', '#00bcd4', '#76c6ba', "#607d8b", '#167a72']
custom_palette = sns.set_palette(sns.color_palette(colors))

def set_group_data(metrics):
    group_by = ast.literal_eval(metrics["group_by"].iloc[0])
    for by_field in group_by:
        value_index = group_by.index(by_field)
        metrics[by_field] = metrics["group_id"].apply(lambda x: ast.literal_eval(x)[value_index])
        if by_field == "model" or by_field == "model_type" or by_field == "sentience":
            metrics[by_field] = metrics[by_field].map(MODEL_NAME_MAP)
    return group_by

def plot_default_metrics(metrics_lst, metric, output_dir, output_format="png", plot_type="bar"): 
    merged_metrics = concat_dfs(metrics_lst)

    if metric not in merged_metrics.columns:
        print(f"Skipping. Metric {metric} not found in the input data.")
        return

    group_by = set_group_data(merged_metrics)

    assert len(group_by) > 1, "Group by field should have at least 2 values."

    hue_attr = group_by[-2]
    x_attr = group_by[-1]

    output_dir = pathlib.Path(output_dir) / f"{output_format}_figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # ax.set_title(metric)
    ax.set_xlabel(LABEL_MAP.get(x_attr, x_attr), size=XLABEL_SIZE, labelpad=50)
    ax.set_ylabel(metric, size=YLABEL_SIZE, labelpad=20)
    ax.tick_params(axis='x', labelsize=TICK_SIZE, rotation=X_ROTATION)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    # ax.title.set_size(TITLE_SIZE)
    
    if plot_type == "bar":
        sns.barplot(data=merged_metrics, x=x_attr, y=metric, hue=hue_attr, ax=ax, err_kws={'linewidth': 5}, palette=custom_palette)
        ax.legend(title=None, title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
    elif plot_type == "violin":
        sns.violinplot(data=merged_metrics, x=x_attr, y=metric, hue=hue_attr, ax=ax, palette=custom_palette)
        ax.legend(title=None, title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)

    plt.tight_layout()

    plot_path = f"{output_dir}/fig_{metric}_by_{x_attr}.{output_format}"
    print(f"Saving figure to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

def plot_global_metrics_n_gram_diversity(metrics_lst, output_dir, output_format="png"):
    merged_metrics = concat_dfs(metrics_lst)
    group_by = set_group_data(merged_metrics)

    output_dir = pathlib.Path(output_dir) / f"{output_format}_figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    hue_attr = group_by[-2]
    group_attr = group_by[-1]

    group_ids = merged_metrics[group_attr].unique()

    metric = "metric_corpus_n_gram_diversity"
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # ax.legend(ncol=5, loc="center left", bbox_to_anchor=(0, 1.05), title=LABEL_MAP.get(hue_attr, hue_attr), title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
    # ax.set_title(metric)
    # ax.title.set_size(TITLE_SIZE)
    ax.set_xlabel("n-gram size", size=XLABEL_SIZE, labelpad=20)
    ax.set_ylabel(metric, size=YLABEL_SIZE, labelpad=20)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    
    all_group_metrics = []
    for group_id in group_ids:
        group_metrics = merged_metrics[merged_metrics[group_attr] == group_id].copy()
        group_metrics["metric_corpus_n_gram_diversity"] = group_metrics["metric_corpus_n_gram_diversity"].apply(lambda x: ast.literal_eval(x))
        group_metrics["n_gram"] = [list(range(1, len(group_metrics["metric_corpus_n_gram_diversity"].iloc[0])+1)) for _ in range(len(group_metrics))]
        group_metrics = group_metrics.explode(["metric_corpus_n_gram_diversity", "n_gram"])
        group_metrics["n_gram"] = group_metrics["n_gram"].astype(str) 
        all_group_metrics.append(group_metrics)
    
    all_group_metrics = pd.concat(all_group_metrics)
    sns.lineplot(data=all_group_metrics, x="n_gram", y=metric, hue=hue_attr, style=group_attr, markers=True, lw=5, ax=ax, palette=custom_palette)
    
    handles, labels = ax.get_legend_handles_labels()
    labels = [LABEL_MAP.get(label, label) for label in labels]
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
    plt.tight_layout()

    plot_path = f"{output_dir}/fig_{metric}.{output_format}"
    print(f"Saving figure to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

def plot_global_metrics_raw_surprises(metrics_lst, output_dir, output_format="png"):
    merged_metrics = concat_dfs(metrics_lst)
    group_by = set_group_data(merged_metrics)

    output_dir = pathlib.Path(output_dir) / f"{output_format}_figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    hue_attr = group_by[-2]
    group_attr = group_by[-1]

    group_ids = merged_metrics[group_attr].unique()

    metric = "metric_avg_raw_surprises"
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # ax.set_title(metric)
    # ax.title.set_size(TITLE_SIZE)
    ax.set_xlabel("Sentence", size=XLABEL_SIZE, labelpad=20)
    ax.set_ylabel(metric, size=YLABEL_SIZE, labelpad=20)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    # ax.legend(ncol=5, loc="center left", bbox_to_anchor=(0, 1.05), title=LABEL_MAP.get(hue_attr, hue_attr), title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)

    all_group_metrics = []
    for group_id in group_ids:
        group_metrics = merged_metrics[merged_metrics[group_attr] == group_id].copy()
        group_metrics["metric_avg_raw_surprises"] = group_metrics["metric_avg_raw_surprises"].apply(lambda x: ast.literal_eval(x))
        max_surprise_len = max([len(x) for x in group_metrics["metric_avg_raw_surprises"]])
        group_metrics["metric_avg_raw_surprises"] = group_metrics["metric_avg_raw_surprises"].apply(lambda x: x + [x[-1]]*(max_surprise_len-len(x)))
        group_metrics["fragment"] = [list(range(1, max_surprise_len+1)) for _ in range(len(group_metrics))]
        group_metrics = group_metrics.explode(["metric_avg_raw_surprises", "fragment"])
        group_metrics["fragment"] = group_metrics["fragment"].astype(str)
        all_group_metrics.append(group_metrics)
    
    all_group_metrics = pd.concat(all_group_metrics)
    sns.lineplot(data=all_group_metrics, x="fragment", y=metric, hue=hue_attr, style=group_attr, markers=True, lw=5, ax=ax, palette=custom_palette, legend="brief")
    handles, labels = ax.get_legend_handles_labels()
    labels = [LABEL_MAP.get(label, label) for label in labels]
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plot_path = f"{output_dir}/fig_{metric}.{output_format}"
    print(f"Saving figure to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

def plot_default_metrics_by_model_size(metrics_lst, metric, output_dir, output_format="png"): 
    merged_metrics = concat_dfs(metrics_lst)

    if metric not in merged_metrics.columns:
        print(f"Skipping. Metric {metric} not found in the input data.")
        return

    # filter out models with unknown size
    merged_metrics = merged_metrics[merged_metrics["model"].isin(MODEL_SIZE_MAP.keys())]
    merged_metrics["model_size"] = merged_metrics["model"].apply(lambda x: MODEL_SIZE_MAP.get(x, 0) / 1e9)
    merged_metrics["model_family"] = merged_metrics["model"].apply(get_model_family)
    group_by = set_group_data(merged_metrics)

    assert len(group_by) > 1, "Group by field should have at least 2 values."

    group_attr = group_by[-1]

    group_ids = merged_metrics[group_attr].unique()
    hue_attr = "model_family"
    x_attr = "model_size"

    default_output_dir = pathlib.Path(output_dir) / f"{output_format}_figs" / "model_size"
    default_output_dir.mkdir(parents=True, exist_ok=True)
    # fig, ax = plt.subplots(figsize=FIG_SIZE)
    # ax.set_xlabel("Model size (in billions)", size=XLABEL_SIZE)
    # ax.set_ylabel(metric, size=YLABEL_SIZE)
    # ax.tick_params(axis='x', labelsize=TICK_SIZE)
    # ax.tick_params(axis='y', labelsize=TICK_SIZE)
    # ax.title.set_size(TITLE_SIZE)
    plt.figure(figsize=FIG_SIZE)
    sns.lmplot(data=merged_metrics, x=x_attr, y=metric, hue=group_attr, legend=False)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1.05))
    # ax.legend(title=None, title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)

    plt.tight_layout()

    plot_path = f"{default_output_dir}/fig_{metric}_by_model_size.{output_format}"
    print(f"Saving figure to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

    # for group_id in group_ids:
    #     group_metrics = merged_metrics[merged_metrics[group_attr] == group_id].copy()

    #     default_output_dir = pathlib.Path(output_dir) / f"{output_format}_figs" / "model_size" / group_id
    #     default_output_dir.mkdir(parents=True, exist_ok=True)
    #     fig, ax = plt.subplots(figsize=FIG_SIZE)
    #     ax.set_title(group_id)
    #     ax.set_xlabel("Model size (in billions)", size=XLABEL_SIZE)
    #     ax.set_ylabel(metric, size=YLABEL_SIZE)
    #     ax.tick_params(axis='x', labelsize=TICK_SIZE)
    #     ax.tick_params(axis='y', labelsize=TICK_SIZE)
    #     ax.title.set_size(TITLE_SIZE)
        
    #     # sns.scatterplot(data=group_metrics, x=x_attr, y=metric, hue=hue_attr, style=hue_attr, s=400, markers=True, ax=ax, palette=custom_palette)
    #     sns.regplot(data=group_metrics, x=x_attr, y=metric, ax=ax)
    #     # sns.lineplot(data=merged_metrics, x=x_attr, y=metric, hue=hue_attr, style=hue_attr, lw=5, markers=True, ax=ax, palette=custom_palette, legend="brief")
    #     ax.legend(title=None, title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
    #     # ax.set_xscale("log")
    #     # sns.lmplot(data=merged_metrics, x=x_attr, y=metric, hue=hue_attr, scatter=True, palette=custom_palette, legend=True)

    #     plt.tight_layout()

    #     plot_path = f"{default_output_dir}/fig_{metric}_by_model_size.{output_format}"
    #     print(f"Saving figure to {plot_path}")
    #     plt.savefig(plot_path)
    #     plt.close()

def plot_default_metrics_by_model_family(metrics_lst, metric, output_dir, output_format="png"): 
    merged_metrics = concat_dfs(metrics_lst)

    if metric not in merged_metrics.columns:
        print(f"Skipping. Metric {metric} not found in the input data.")
        return

    merged_metrics["model_family"] = merged_metrics["model"].apply(get_model_family)
    group_by = set_group_data(merged_metrics)

    assert len(group_by) > 1, "Group by field should have at least 2 values."

    group_attr = group_by[-1]

    group_ids = merged_metrics[group_attr].unique()
    human_metrics = merged_metrics[merged_metrics["model_family"] == "human"]
    model_metrics = merged_metrics[merged_metrics["model_family"] != "human"]
    hue_attr = "model_family"
    x_attr = group_by[-1]

    for group_id in group_ids:
        human_group_metrics = human_metrics[human_metrics[group_attr] == group_id].copy()
        model_group_metrics = model_metrics[model_metrics[group_attr] == group_id].copy()

        default_output_dir = pathlib.Path(output_dir) / f"{output_format}_figs" / "model_family" / group_id
        default_output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        # ax.set_title(metric)
        ax.set_xlabel(LABEL_MAP.get(x_attr, x_attr), size=XLABEL_SIZE)
        ax.set_ylabel(metric, size=YLABEL_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        # ax.title.set_size(TITLE_SIZE)
        
        sns.swarmplot(data=merged_metrics, x=x_attr, y=metric, hue=hue_attr, size=10, ax=ax, palette=custom_palette)
        # ax.legend(title=None, title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)
        ax.legend(ncol=5, loc="center left", bbox_to_anchor=(0, 1.05), title=None, title_fontsize=LEGEND_TITLE_SIZE, fontsize=LEGEND_SIZE)

        plt.tight_layout()

        plot_path = f"{default_output_dir}/fig_{metric}_by_model_family.{output_format}"
        print(f"Saving figure to {plot_path}")
        plt.savefig(plot_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-files", type=str, nargs="+", help="Path to csv metrics file(s)", required=True)
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory to save plots", default=None)
    parser.add_argument("-f", "--output-format", type=str, choices=["png", "pdf", "svg"], default="png", help="Format to save the plots in.")
    parser.add_argument("-p", "--plots", type=str, nargs="+", choices=["default", "default_model_size", "default_model_family", "n_gram_diversity", "raw_surprises"], default=["default"], help="Plots to generate")
    parser.add_argument("-t", "--plot-type", type=str, choices=["bar", "violin"], default="violin", help="Type of plot to generate")

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

    for plot in args.plots:
        if plot.startswith("default"):
            for col in metrics_lst[0].columns:
                if col.startswith("metric_") and is_numeric_dtype(metrics[col].iloc[0]):
                    if plot == "default":
                        plot_default_metrics(metrics_lst, col, output_dir, output_format=args.output_format, plot_type=args.plot_type)
                    if plot == "default_model_size":
                        plot_default_metrics_by_model_size(metrics_lst, col, output_dir, output_format=args.output_format)
                    if plot == "default_model_family":
                        plot_default_metrics_by_model_family(metrics_lst, col, output_dir, output_format=args.output_format)
        elif plot == "n_gram_diversity":
            plot_global_metrics_n_gram_diversity(metrics_lst, output_dir, output_format=args.output_format)
        elif plot == "raw_surprises":
            plot_global_metrics_raw_surprises(metrics_lst, output_dir, output_format=args.output_format)
        else:
            print(f"Plot {plot} not supported. Skipping...")

if __name__ == "__main__":
    main()
import argparse
from tqdm import tqdm
import pathlib
from collections import defaultdict
from statistics import mean
import random

from utils import read_json, write_json, find_files, compute_usage
from metrics import compute_inverse_homogenization, compute_novelty, compute_theme_uniqueness, compute_dsi, compute_surprise, compute_n_gram_diversity, get_words, get_sentences

DEF_EMB_MODEL = "thenlper/gte-large"
DEF_EMB_TYPE = "sentence_embedding"
DEF_DIST_FN = "cosine"
DEF_SPACY_LANG = "en_core_web_sm"
DEF_PREPROCESSING_ARGS = {
    "lower": True,
    "remove_punct": True,
    "remove_stopwords": True,
    "lemmatize": True,
    "dominant_k": None,
    "unique": True
}

def compute_metrics(results, config):
    metrics = {}

    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }

    cost = {
        "input": 0,
        "output": 0,
        "total": 0
    }

    response_attr = "output"

    preprocessing_args = {
        "lower": config["lower"],
        "remove_punct": config["remove_punct"],
        "remove_stopwords": config["remove_stopwords"],
        "lemmatize": config["lemmatize"],
        "dominant_k": config["dominant_k"],
        "unique": config["unique"]
    }

    results = [result for result in results if response_attr in result and result[response_attr]]

    stories = [result[response_attr] for result in results]

    if len(stories) > 1:
        print("Computing global metrics")
        inv_homogen = compute_inverse_homogenization(stories, config["emb_model"], config["emb_type"], config["emb_strategy"], config["distance_fn"], preprocessing_args)
        novelty = compute_novelty(stories, config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
        theme_uniqueness = compute_theme_uniqueness(stories, config["emb_model"], config["emb_type"], config["emb_strategy"], config["cluster_linkage"], config["cluster_dist_threshold"], preprocessing_args)
        corpus_dsi = compute_dsi("".join(stories), config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
        corpus_n_gram_diversity, corpus_n_gram_frequency = compute_n_gram_diversity("".join(stories), config["max_n_gram"])

    for result_idx, result in tqdm(enumerate(results), total=len(results), desc="Computing local metrics"):
        result["metrics"] = {}

        if response_attr in result:
            all_words = get_words(result[response_attr], lower=False, remove_punct=True, remove_stopwords=False, lemmatize=False, unique=False, dominant_k=None)
            unique_words = list(set([word.lower() for word in all_words if all_words]))
            concepts = get_words(result[response_attr], lower=True, remove_punct=True, remove_stopwords=True, lemmatize=True, unique=True, dominant_k=None)
            sentences = get_sentences(result[response_attr])
            sentence_words = [get_words(sentence, lower=False, remove_punct=True, remove_stopwords=False, lemmatize=False, unique=False, dominant_k=None) for sentence in sentences]
            sentence_unique_words = [list(set([word.lower() for word in words])) for words in sentence_words]

            # basic metrics
            result["metrics"]["length_in_chars"] = len(result[response_attr])
            result["metrics"]["length_in_words"] = len(all_words)
            result["metrics"]["length_in_unique_words"] = len(unique_words)
            result["metrics"]["length_in_concepts"] = len(concepts)
            result["metrics"]["length_in_sentences"] = len(sentences)
            result["metrics"]["avg_word_length_in_chars"] = mean([len(word) for word in all_words])
            result["metrics"]["avg_sentence_length_in_chars"] = mean([len(sentence) for sentence in sentences])
            result["metrics"]["avg_sentence_length_in_words"] = mean([len(words) for words in sentence_words])
            result["metrics"]["avg_sentence_length_in_unique_words"] = mean([len(words) for words in sentence_unique_words])
            
            # complex metrics
            result["metrics"]["dsi"] = compute_dsi(result[response_attr], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
            result["metrics"]["surprise"] = compute_surprise(result[response_attr], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
            result["metrics"]["n_gram_diversity"], _ = compute_n_gram_diversity(result[response_attr], config["max_n_gram"])
            
            if len(stories) > 1:
                result["metrics"]["inv_homogen"] = inv_homogen[result_idx]
                result["metrics"]["novelty"] = novelty[result_idx]
                result["metrics"]["theme_uniqueness"] = theme_uniqueness[result_idx]

            if config["report_usage"]:
                sample_usage, sample_cost = compute_usage(result, result["metadata"]["model"])

                if sample_usage:
                    usage["input_tokens"] += sample_usage["input_tokens"]
                    usage["output_tokens"] += sample_usage["output_tokens"]
                    usage["total_tokens"] += sample_usage["input_tokens"] + sample_usage["output_tokens"]

                if sample_cost:
                    cost["input"] += sample_cost["input"]
                    cost["output"] += sample_cost["output"]
                    cost["total"] += sample_cost["total"]

    metrics["num_samples"] = len(results)

    # basic metrics
    metrics["avg_length_in_chars"] = mean([result["metrics"]["length_in_chars"] for result in results])
    metrics["avg_length_in_words"] = mean([result["metrics"]["length_in_words"] for result in results])
    metrics["avg_length_in_unique_words"] = mean([result["metrics"]["length_in_unique_words"] for result in results])
    metrics["avg_length_in_concepts"] = mean([result["metrics"]["length_in_concepts"] for result in results])
    metrics["avg_length_in_sentences"] = mean([result["metrics"]["length_in_sentences"] for result in results])
    metrics["avg_word_length_in_chars"] = mean([result["metrics"]["avg_word_length_in_chars"] for result in results])
    metrics["avg_sentence_length_in_chars"] = mean([result["metrics"]["avg_sentence_length_in_chars"] for result in results])
    metrics["avg_sentence_length_in_words"] = mean([result["metrics"]["avg_sentence_length_in_words"] for result in results])
    metrics["avg_sentence_length_in_unique_words"] = mean([result["metrics"]["avg_sentence_length_in_unique_words"] for result in results])

    # complex metrics
    metrics["avg_dsi"] = mean([result["metrics"]["dsi"] for result in results])
    metrics["avg_surprise"] = mean([result["metrics"]["surprise"] for result in results])
    metrics["avg_n_gram_diversity"] = [mean([result["metrics"]["n_gram_diversity"][n_gram_len-1] for result in results]) for n_gram_len in range(1, config["max_n_gram"]+1)]
    
    if len(stories) > 1:
        metrics["avg_inv_homogen"] = mean([result["metrics"]["inv_homogen"] for result in results])
        metrics["avg_novelty"] = mean([result["metrics"]["novelty"] for result in results])
        metrics["avg_theme_uniqueness"] = mean([result["metrics"]["theme_uniqueness"] for result in results])
        metrics["corpus_dsi"] = corpus_dsi
        metrics["corpus_n_gram_diversity"] = corpus_n_gram_diversity
        metrics["num_unique_stories"] = len(set(stories))
        metrics[f"top_{config['max_frequency']}_n_grams"] = {}

        for i, freq_counter in enumerate(corpus_n_gram_frequency):
            metrics[f"top_{config['max_frequency']}_n_grams"][f"{i+1}-gram"] = {" ".join(freq[0]): freq[1] for freq in freq_counter.most_common(config["max_frequency"])}

    if config["report_usage"]:
        metrics["usage"] = usage
        metrics["cost"] = cost

    return metrics

def set_metadata(results, results_file):
    for sample in results["data"]:
        sample["metadata"] = {
            "source": results_file,
            "model": results["metadata"]["model"],
            "model_args": results["metadata"].get("model_args")
        }

        if sample["metadata"]["model"] == "human":
            sample["metadata"]["participant_id"] = results["metadata"]["participant_id"]

        model_path = results["metadata"].get("model_path")
        tokenizer_path = results["metadata"].get("tokenizer_path")

        if model_path:
            sample["metadata"]["model_path"] = model_path
        if tokenizer_path:
            sample["metadata"]["tokenizer_path"] = tokenizer_path

    return results

def group_results_by_id(results):
    result_groups = defaultdict(list)

    for result in results:
        result_groups[result["id"]].append(result)

    return result_groups

def deduplicate_results(results, by="output"):
    value_set = set()
    unique_results = []
    
    for res in results:
        value = res[by]
        if value not in value_set:
            unique_results.append(res)
            value_set.add(value)

    return unique_results

def report_metrics(results_files, config):
    grouped_results = defaultdict(list)
    all_results = []

    for results_file in results_files:
        results = read_json(results_file)
        
        try:
            if "data" in results:
                results = set_metadata(results, results_file)
                all_results.extend(results["data"])
        except Exception as e:
            print(results_file)
            raise e
    
    grouped_results = group_results_by_id(all_results)

    output_dir = pathlib.Path(config["output_dir"]) if config["output_dir"] else pathlib.Path(results_files[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = config["num_samples"]

    if config["deduplicate"]:
        for group_id, group_results in grouped_results.items():
            grouped_results[group_id] = deduplicate_results(group_results, by=config["deduplicate_by"])
        
        min_group_size = min([len(group_results) for group_results in grouped_results.values()])

        if num_samples:
            num_samples = min(num_samples, min_group_size)
    
    for group_id, group_results in grouped_results.items():    
        if num_samples:
            group_results = random.sample(group_results, min(num_samples, len(group_results)))
        
        metrics = compute_metrics(group_results, config)
        
        results = {
            "metadata": {
                "source": results_files,
                "group_id": group_id,
                "size": len(group_results)
            },
            "metrics": metrics,
            "data": group_results
        }
        results_file = output_dir / f"{group_id}_results.json"
        write_json(results, results_file)
    
    write_json(config, output_dir / "config.json")

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(prog="report_metrics.py", description="Report evaluation metrics")
    parser.add_argument("-r", "--results-path", type=str, help="Path to evaluation results file in json or directory", required=True)
    parser.add_argument("-u", "--report-usage", action="store_true", help="Report usage metrics", default=True)
    parser.add_argument("-sl", "--spacy-lang", type=str, help="Spacy language model", default=DEF_SPACY_LANG)
    parser.add_argument("-ng", "--max-n-gram", type=int, help="Maximum n-gram to consider", default=5)
    parser.add_argument("-o", "--output-dir", type=str, help="Output dir to save the output files", default=None)
    parser.add_argument("-c", "--config", type=str, help="Path to config file", default=None)
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to consider", default=None)
    parser.add_argument("-de", "--deduplicate", action="store_true", help="Remove duplicate samples")
    parser.add_argument("-deby", "--deduplicate-by", type=str, help="Remove duplicate samples by attribute", default="output")
    parser.add_argument("-mf", "--max-frequency", type=int, help="Maximum n-gram frequency to consider", default=10)

    emb_group = parser.add_argument_group("Embedding arguments")
    emb_group.add_argument("-em", "--emb-model", type=str, help="Sentence embedding model", default=DEF_EMB_MODEL)
    emb_group.add_argument("-et", "--emb-type", type=str, help="Embedding type", default=DEF_EMB_TYPE)
    emb_group.add_argument("-es", "--emb-strategy", type=str, help="Embedding strategy", default="direct")
    emb_group.add_argument("-d", "--distance-fn", type=str, help="Distance function", default=DEF_DIST_FN)

    pp_group = parser.add_argument_group("Preprocessing arguments")
    pp_group.add_argument("-l", "--lower", action="store_true", help="Lowercase text")
    pp_group.add_argument("-rp", "--remove-punct", action="store_true", help="Remove punctuation")
    pp_group.add_argument("-rs", "--remove-stopwords", action="store_true", help="Remove stopwords")
    pp_group.add_argument("-lm", "--lemmatize", action="store_true", help="Lemmatize text")
    pp_group.add_argument("-dk", "--dominant-k", type=none_or_int, help="Number of dominant words to consider", default=None)
    pp_group.add_argument("-q", "--unique", action="store_true", help="Unique words")

    cl_group = parser.add_argument_group("Clustering arguments")
    cl_group.add_argument("-cl", "--cluster-linkage", type=str, help="Cluster linkage", default="ward")
    cl_group.add_argument("-cdt", "--cluster-dist-threshold", type=float, help="Cluster distance threshold", default=0.5)
    
    args = parser.parse_args()

    print(f"Reporting metrics for: {args.results_path}")

    files_to_process = []

    results_path = pathlib.Path(args.results_path)

    if results_path.is_file():
        files_to_process.append(args.results_path)
    else:
        files_to_process.extend(find_files(args.results_path))

    config = vars(args)

    if args.config:
        config.update(read_json(args.config))

    report_metrics(files_to_process, config)

if __name__ == "__main__":
    main()
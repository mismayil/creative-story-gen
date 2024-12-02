import argparse
from tqdm import tqdm
import pathlib
from collections import defaultdict
from statistics import mean
import random

from utils import read_json, write_json, find_files, compute_usage
from metrics import (compute_inverse_homogenization, compute_novelty, 
                     compute_theme_uniqueness, compute_dsi, compute_surprise, 
                     compute_n_gram_diversity, get_words, get_sentences, 
                     compute_pos_diversity, compute_dependency_complexity, 
                     compute_pos_complexity, compute_constituency_complexity,
                     compute_flesch_readability_scores, compute_interestingness)

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

MODEL_TYPES = ["gpt-4", "claude-3-5", "gemini-1.5", "llama-3.1", "human"]

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

    output_attr = config["output_attr"]

    preprocessing_args = {
        "lower": config["lower"],
        "remove_punct": config["remove_punct"],
        "remove_stopwords": config["remove_stopwords"],
        "lemmatize": config["lemmatize"],
        "dominant_k": config["dominant_k"],
        "unique": config["unique"]
    }

    results = [result for result in results if output_attr in result and result[output_attr]]

    stories = [result[output_attr] for result in results]

    if len(stories) > 1:
        print("Computing global metrics")
        inv_homogen = compute_inverse_homogenization(stories, config["emb_model"], config["emb_type"], config["emb_strategy"], config["distance_fn"], preprocessing_args)
        novelty = compute_novelty(stories, config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
        theme_uniqueness, theme_clusters = compute_theme_uniqueness(stories, config["emb_model"], config["emb_type"], config["emb_strategy"], config["cluster_linkage"], config["cluster_dist_threshold"], preprocessing_args)
        corpus_dsi = compute_dsi("".join(stories), config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
        corpus_n_gram_diversity, corpus_n_gram_frequency = compute_n_gram_diversity("".join(stories), config["max_n_gram"])
        corpus_pos_diversity, corpus_pos_frequency = compute_pos_diversity("".join(stories), config["max_n_gram"])

    for result_idx, result in tqdm(enumerate(results), total=len(results), desc="Computing local metrics"):
        result["metrics"] = {}

        if output_attr in result:
            all_words = get_words(result[output_attr], lower=False, remove_punct=True, remove_stopwords=False, lemmatize=False, unique=False, dominant_k=None)
            unique_words = list(set([word.lower() for word in all_words if all_words]))
            concepts = get_words(result[output_attr], lower=True, remove_punct=True, remove_stopwords=True, lemmatize=True, unique=True, dominant_k=None)
            sentences = get_sentences(result[output_attr])
            sentence_words = [get_words(sentence, lower=False, remove_punct=True, remove_stopwords=False, lemmatize=False, unique=False, dominant_k=None) for sentence in sentences]
            sentence_unique_words = [list(set([word.lower() for word in words])) for words in sentence_words]

            # basic metrics
            result["metrics"]["length_in_chars"] = len(result[output_attr])
            result["metrics"]["length_in_words"] = len(all_words)
            result["metrics"]["length_in_unique_words"] = len(unique_words)
            result["metrics"]["length_in_concepts"] = len(concepts)
            result["metrics"]["length_in_sentences"] = len(sentences)
            result["metrics"]["type_token_ratio"] = len(unique_words) / len(all_words) if all_words else 0
            result["metrics"]["avg_word_length_in_chars"] = mean([len(word) for word in all_words])
            result["metrics"]["avg_sentence_length_in_chars"] = mean([len(sentence) for sentence in sentences])
            result["metrics"]["avg_sentence_length_in_words"] = mean([len(words) for words in sentence_words])
            result["metrics"]["avg_sentence_length_in_unique_words"] = mean([len(words) for words in sentence_unique_words])
            result["metrics"]["length_in_first_person_singular"] = len([word for word in all_words if word.lower() in ["i", "me", "my", "mine", "myself"]])
            result["metrics"]["length_in_first_person_plural"] = len([word for word in all_words if word.lower() in ["we", "us", "our", "ours", "ourselves"]])
            result["metrics"]["length_in_second_person"] = len([word for word in all_words if word.lower() in ["you", "your", "yours", "yourself", "yourselves"]])
            result["metrics"]["length_in_third_person_singular"] = len([word for word in all_words if word.lower() in ["he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself"]])
            result["metrics"]["length_in_third_person_plural"] = len([word for word in all_words if word.lower() in ["they", "them", "their", "theirs", "themselves"]])
            result["metrics"]["length_in_first_person"] = result["metrics"]["length_in_first_person_singular"] + result["metrics"]["length_in_first_person_plural"]
            result["metrics"]["length_in_third_person"] = result["metrics"]["length_in_third_person_singular"] + result["metrics"]["length_in_third_person_plural"]

            
            # complex metrics
            result["metrics"]["dsi"] = compute_dsi(result[output_attr], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
            surprise, raw_surprises = compute_surprise(result[output_attr], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
            result["metrics"]["surprise"] = surprise
            result["metrics"]["raw_surprises"] = raw_surprises
            result["metrics"]["n_gram_diversity"], _ = compute_n_gram_diversity(result[output_attr], config["max_n_gram"])
            result["metrics"]["pos_diversity"], _ = compute_pos_diversity(result[output_attr], config["max_n_gram"])
            
            dependency_paths, dependency_num_clauses = compute_dependency_complexity(result[output_attr])
            result["metrics"]["avg_dep_num_clauses"] = mean(dependency_num_clauses)
            result["metrics"]["max_dep_num_clauses"] = max(dependency_num_clauses)
            result["metrics"]["avg_dep_path_length"] = mean([mean([len(path) for path, freq in path_counter.items()]) for path_counter in dependency_paths])
            result["metrics"]["max_dep_path_length"] = max([max([len(path) for path, freq in path_counter.items()]) for path_counter in dependency_paths])
            
            constituency_complexity = compute_constituency_complexity(result[output_attr])
            result["metrics"]["avg_constituency_tree_depth"] = mean(constituency_complexity)
            result["metrics"]["max_constituency_tree_depth"] = max(constituency_complexity)

            flesch_ease, flesch_kincaid = compute_flesch_readability_scores(result[output_attr])
            result["metrics"]["readability_flesch_ease"] = flesch_ease
            result["metrics"]["readability_flesch_kincaid"] = flesch_kincaid

            if "boring_theme" in result and "summary" in result:
                result["metrics"]["interestingness"] = compute_interestingness(result["summary"], result["boring_theme"], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)

            pos_complexity = compute_pos_complexity(result[output_attr])
            for pos, pos_comps in pos_complexity.items():
                result["metrics"][f"avg_pos_{pos.lower()}_ratio"] = mean(pos_comps) if pos_comps else 0

            if len(stories) > 1:
                result["metrics"]["inv_homogen"] = inv_homogen[result_idx]
                result["metrics"]["novelty"] = novelty[result_idx]
                result["metrics"]["theme_uniqueness"] = theme_uniqueness[result_idx]
                result["metrics"]["theme_cluster"] = int(theme_clusters[result_idx])

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
    metrics["avg_raw_surprises"] = [mean([result["metrics"]["raw_surprises"][surprise_idx] for result in results if "metrics" in result and surprise_idx < len(result["metrics"]["raw_surprises"])]) for surprise_idx in range(max([len(r["metrics"]["raw_surprises"]) for r in results if "metrics" in r]))]
    metrics["avg_n_gram_diversity"] = [mean([result["metrics"]["n_gram_diversity"][n_gram_len-1] for result in results]) for n_gram_len in range(1, config["max_n_gram"]+1)]
    metrics["avg_pos_diversity"] = [mean([result["metrics"]["pos_diversity"][n_gram_len-1] for result in results]) for n_gram_len in range(1, config["max_n_gram"]+1)]
    metrics["avg_dep_num_clauses"] = mean([result["metrics"]["avg_dep_num_clauses"] for result in results])
    metrics["avg_max_dep_num_clauses"] = mean([result["metrics"]["max_dep_num_clauses"] for result in results])
    metrics["avg_dep_path_length"] = mean([result["metrics"]["avg_dep_path_length"] for result in results])
    metrics["avg_max_dep_path_length"] = mean([result["metrics"]["max_dep_path_length"] for result in results])
    
    if len(stories) > 1:
        metrics["avg_inv_homogen"] = mean([result["metrics"]["inv_homogen"] for result in results])
        metrics["avg_novelty"] = mean([result["metrics"]["novelty"] for result in results])
        metrics["avg_theme_uniqueness"] = mean([result["metrics"]["theme_uniqueness"] for result in results])
        metrics["corpus_dsi"] = corpus_dsi
        metrics["corpus_n_gram_diversity"] = corpus_n_gram_diversity
        metrics["corpus_pos_diversity"] = corpus_pos_diversity
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

def group_results_by_attr(results: list, attr="id"):
    result_groups = defaultdict(list)

    for result in results:
        result_groups[result[attr]].append(result)

    return result_groups

def group_results_by_model(results: list):
    result_groups = defaultdict(list)

    for result in results:
        result_groups[result["metadata"]["model"]].append(result)

    return result_groups

def group_results_by_model_type(results: list):
    result_groups = defaultdict(list)

    for result in results:
        model = result["metadata"]["model"]
        for model_type in MODEL_TYPES:
            if model.startswith(model_type):
                result_groups[model_type].append(result)
                break

    return result_groups

def group_results_by_sentience(results: list):
    result_groups = defaultdict(list)

    for result in results:
        model = result["metadata"]["model"]
        if model == "human":
            result_groups["human"].append(result)
        else:
            result_groups["machine"].append(result)

    return result_groups

def group_results_by(results: list, by="id"):
    if by == "model":
        return group_results_by_model(results)
    elif by == "sentience":
        return group_results_by_sentience(results)
    elif by == "model_type":
        return group_results_by_model_type(results)
    else:
        return group_results_by_attr(results, attr=by)

def group_results(results, by=["id"]):
    for b in by:
        print(f"Grouping results by {b}")
        if isinstance(results, list):
            results = group_results_by(results, by=b)
        else:
            new_results = {}
            for key, value in results.items():
                subresults = group_results_by(value, by=b)
                for subkey, subvalue in subresults.items():
                    if isinstance(key, tuple):
                        new_key = (key + (subkey,))
                    else:
                        new_key = (key, subkey)
                    new_results[new_key] = subvalue
            results = new_results
    return results

def deduplicate_results(results, by="output"):
    value_set = set()
    unique_results = []
    
    for res in results:
        value = res[by]
        if value not in value_set:
            unique_results.append(res)
            value_set.add(value)

    return unique_results

def filter_results(results, by="sentence_length", value=(3, 6), output_attr="output"):
    filtered_results = []

    for result in tqdm(results, total=len(results), desc=f"Filtering results by {by}"):
        if by == "sentence_length":
            sentences = get_sentences(result[output_attr])
            min_value = float("-inf")
            max_value = float("inf")

            if len(value) == 1:
                max_value = value[0]
            elif len(value) == 2:
                min_value, max_value = value
            else:
                raise ValueError(f"Invalid filter value: {value}")

            if min_value <= len(sentences) <= max_value:
                filtered_results.append(result)
        else:
            raise ValueError(f"Invalid filter attribute: {by}")
    
    return filtered_results

def report_metrics(results_files, config):
    grouped_results = defaultdict(list)
    all_results = []
    num_result_files = 0

    for results_file in results_files:
        results = read_json(results_file)
        
        try:
            if "data" in results:
                results = set_metadata(results, results_file)
                all_results.extend(results["data"])
                num_result_files += 1

                if config["verbose"]:
                    print(f"Number of samples for {results['metadata']['model']}: {len(results['data'])}")
        except Exception as e:
            print(results_file)
            raise e
    
    print(f"Number of result files: {num_result_files}")
    print(f"Number of total samples before filtering: {len(all_results)}")

    filter_by = config.get("filter_by")

    if filter_by:
        for filter_by_attr, filter_by_value in filter_by.items():
            all_results = filter_results(all_results, by=filter_by_attr, value=filter_by_value, output_attr=config["output_attr"])

    grouped_results = group_results(all_results, by=config["group_by"])
    
    # print stats
    print(f"Number of total samples: {len(all_results)}")
    print(f"Number of groups: {len(grouped_results)}")
    print("Groups:")
    for group_id, group_res in grouped_results.items():
        print(f"\tGroup {group_id}: {len(group_res)} samples")

    output_dir = pathlib.Path(config["output_dir"]) if config["output_dir"] else pathlib.Path(results_files[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["deduplicate"]:
        print("Deduplicating results")
        for group_id, group_res in grouped_results.items():
            grouped_results[group_id] = deduplicate_results(group_res, by=config["deduplicate_by"])
    
    num_samples = config["num_samples"]
    min_group_size = min([len(group_res) for group_res in grouped_results.values()])

    if num_samples:
        num_samples = min(num_samples, min_group_size)
    else:
        num_samples = min_group_size
    
    print(f"Final number of samples: {num_samples}")

    for group_id, group_res in grouped_results.items():
        if config["max_sampling"]:
            group_res = random.sample(group_res, len(group_res))    
        elif num_samples:
            group_res = random.sample(group_res, min(num_samples, len(group_res)))
        
        print(f"Computing metrics for group {group_id}")
        metrics = compute_metrics(group_res, config)
        
        parent_group_ids = []
        leaf_group_id = group_id

        if isinstance(group_id, tuple):
            parent_group_ids = group_id[:-1]
            leaf_group_id = group_id[-1]

        group_output_dir = output_dir

        for parent_group_id in parent_group_ids:
            group_output_dir = group_output_dir / parent_group_id
            
        group_output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "metadata": {
                "group_id": list(group_id) if isinstance(group_id, tuple) else [group_id],
                "config": config
            },
            "metrics": metrics,
            "data": group_res
        }
        
        results_file = group_output_dir / f"{leaf_group_id}_results.json"
        write_json(results, results_file)

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(prog="report_metrics.py", description="Report evaluation metrics")
    parser.add_argument("-r", "--results-paths", type=str, nargs="+", help="Path(s) to evaluation results file in json or directory", required=True)
    parser.add_argument("-u", "--report-usage", action="store_true", help="Report usage metrics", default=True)
    parser.add_argument("-sl", "--spacy-lang", type=str, help="Spacy language model", default=DEF_SPACY_LANG)
    parser.add_argument("-ng", "--max-n-gram", type=int, help="Maximum n-gram to consider", default=5)
    parser.add_argument("-o", "--output-dir", type=str, help="Output dir to save the output files", default=None)
    parser.add_argument("-c", "--config", type=str, help="Path to config file", default=None)
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to consider", default=None)
    parser.add_argument("-de", "--deduplicate", action="store_true", help="Remove duplicate samples")
    parser.add_argument("-deby", "--deduplicate-by", type=str, help="Remove duplicate samples by attribute", default="output")
    parser.add_argument("-mf", "--max-frequency", type=int, help="Maximum n-gram frequency to consider", default=10)
    parser.add_argument("-gb", "--group-by", type=str, nargs="+", help="Group results by attribute(s)", default=["model", "id"])
    parser.add_argument("-oa", "--output-attr", type=str, help="Output attribute", default="output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-ms", "--max-sampling", action="store_true", help="Maximum sampling mode")

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

    print(f"Reporting metrics for: {args.results_paths}")

    files_to_process = []

    for results_path in args.results_paths:
        results_path = pathlib.Path(results_path)

        if results_path.is_file():
            files_to_process.append(results_path)
        else:
            files_to_process.extend(find_files(results_path))

    config = vars(args)

    if args.config:
        config.update(read_json(args.config))

    report_metrics(files_to_process, config)

if __name__ == "__main__":
    main()
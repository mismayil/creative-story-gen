import argparse
from tqdm import tqdm
import pathlib
import spacy
from collections import Counter, defaultdict
from statistics import mean
import random

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score, euclidean_sim, manhattan_sim
from sklearn.cluster import AgglomerativeClustering

from utils import read_json, write_json, find_files, compute_usage, cache

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

SPACY_ENGINE_CACHE = {}
EMB_MODEL_CACHE = {}
EMBEDDING_CACHE = {}
SPACY_CACHE = {}

@cache(cache_dict=SPACY_ENGINE_CACHE)
def load_spacy_engine(language=DEF_SPACY_LANG):
    print(f"Loading spacy engine: {language}")
    return spacy.load(language)

@cache(cache_dict=EMB_MODEL_CACHE)
def load_emb_model(model=DEF_EMB_MODEL):
    print(f"Loading embedding model: {model}")
    return SentenceTransformer(model)

@cache(cache_dict=EMBEDDING_CACHE)
def get_embedding(text, model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE):
    emb_model = load_emb_model(model)
    
    output_value = "sentence_embedding"

    if "token" in emb_type:
        output_value = "token_embeddings"

    embeddings = emb_model.encode(text, output_value=output_value)
    
    if embeddings.ndim == 2:
        embeddings = embeddings.mean(axis=0)

    return embeddings

@cache(cache_dict=SPACY_CACHE)
def get_spacy_doc(text):
    spacy_engine = load_spacy_engine()
    return spacy_engine(text)

def compute_sem_dis(emb1, emb2, distance_fn=DEF_DIST_FN):
    if distance_fn == "cosine":
        return (1 - cos_sim(emb1, emb2)).item()
    elif distance_fn == "dot":
        return (1 - dot_score(emb1, emb2)).item()
    elif distance_fn == "euclidean":
        return (-euclidean_sim(emb1, emb2)).item()
    elif distance_fn == "manhattan":
        return (-manhattan_sim(emb1, emb2)).item()
    else:
        raise ValueError(f"Invalid distance function: {distance_fn}")

def get_sentences(text):
    doc = get_spacy_doc(text)
    return [sent.text for sent in doc.sents]

def get_words(text, lower=True, remove_punct=True, remove_stopwords=True, lemmatize=True, unique=True, dominant_k=None):
    doc = get_spacy_doc(text)
    tokens = [token for token in doc]

    if remove_punct:
        tokens = [token for token in tokens if not token.is_punct]
    
    if remove_stopwords:
        tokens = [token for token in tokens if not token.is_stop]

    words = [token.text for token in tokens]

    if lemmatize:
        words = [token.lemma_ for token in tokens]
    
    if lower:
        words = [word.lower() for word in words]
    
    if dominant_k is None or dominant_k == 0 or dominant_k >= len(words):
        if unique:
            return list(set(words))
        return words

    word_freq = Counter(words)

    return [w[0] for w in word_freq.most_common(dominant_k)]

def compute_story_embedding(story, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct",
                            preprocessing_args=DEF_PREPROCESSING_ARGS):
    if emb_strategy == "direct":
        return get_embedding(story, emb_model, emb_type)
    elif emb_strategy == "by_word":
        words = get_words(story, **preprocessing_args)
        return get_embedding(words, emb_model, emb_type)
    elif emb_strategy == "by_sentence":
        sentences = get_sentences(story)
        return get_embedding(sentences, emb_model, emb_type)
    else:
        raise ValueError(f"Invalid embedding strategy: {emb_strategy}")

def compute_avg_pairwise_distances(embeddings, distance_fn=DEF_DIST_FN):
    if len(embeddings) <= 1:
        return [0]

    avg_pairwise_distances = []
    for i in range(len(embeddings)):
        pairwise_distances = []
        for j in range(len(embeddings)):
            if i != j:
                distance = compute_sem_dis(embeddings[i], embeddings[j], distance_fn)
                pairwise_distances.append(distance)
        avg_pairwise_distances.append(mean(pairwise_distances))
    return avg_pairwise_distances

def compute_avg_sem_dis(text, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    words = get_words(text, **preprocessing_args)
    embeddings = [get_embedding(word, emb_model, emb_type) for word in words]
    return mean(compute_avg_pairwise_distances(embeddings, distance_fn))

def compute_inverse_homogenization(stories, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct", 
                                   distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    story_embeddings = [compute_story_embedding(story, emb_model, emb_type, emb_strategy, preprocessing_args) for story in stories]
    return compute_avg_pairwise_distances(story_embeddings, distance_fn)

def compute_novelty(stories, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, 
                    distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    corpus = "".join(stories)
    corpus_avg_sem_dis = compute_avg_sem_dis(corpus, emb_model, emb_type, distance_fn, preprocessing_args)

    novelty_scores = []

    for story in stories:
        story_avg_sem_dis = compute_avg_sem_dis(story, emb_model, emb_type, distance_fn, preprocessing_args)
        novelty_scores.append(2 * abs(story_avg_sem_dis - corpus_avg_sem_dis))
    
    return novelty_scores

def compute_theme_uniqueness(stories, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct",
                             cluster_linkage="ward", cluster_distance_threshold=0.5, preprocessing_args=DEF_PREPROCESSING_ARGS):
    story_embeddings = [compute_story_embedding(story, emb_model, emb_type, emb_strategy, preprocessing_args) for story in stories]
    clustering = AgglomerativeClustering(n_clusters=None, linkage=cluster_linkage, distance_threshold=cluster_distance_threshold)
    cluster_labels = clustering.fit_predict(story_embeddings)
    cluster_freq = Counter(cluster_labels)
    return [1 / cluster_freq[cluster_labels[i]] for i in range(len(stories))]

def compute_dsi(story, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    return compute_avg_sem_dis(story, emb_model, emb_type, distance_fn, preprocessing_args)

def compute_surprise(story, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    sentences = get_sentences(story)

    if len(sentences) <= 1:
        return 0

    sentence_avg_sem_distances = [compute_avg_sem_dis(sentence, emb_model, emb_type, distance_fn, preprocessing_args) for sentence in sentences]
    raw_surprises = [abs(sentence_avg_sem_distances[i] - sentence_avg_sem_distances[i - 1]) for i in range(1, len(sentence_avg_sem_distances))]

    return (2 / len(raw_surprises)) * sum(raw_surprises)

def compute_n_gram_diversity(story, max_n_gram=5):
    words = get_words(story, remove_punct=False, remove_stopwords=False, lemmatize=False, unique=False)
    all_n_grams = []

    for n in range(1, max_n_gram + 1):
        all_n_grams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    
    all_n_gram_freqs = [Counter(n_grams) for n_grams in all_n_grams]
    n_gram_diversity = [len(n_gram_freqs) / len(n_grams) for n_grams, n_gram_freqs in zip(all_n_grams, all_n_gram_freqs)]

    return n_gram_diversity

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
            
            # semantic metrics
            result["metrics"]["dsi"] = compute_dsi(result[response_attr], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
            result["metrics"]["surprise"] = compute_surprise(result[response_attr], config["emb_model"], config["emb_type"], config["distance_fn"], preprocessing_args)
            result["metrics"]["n_gram_diversity"] = compute_n_gram_diversity(result[response_attr], config["max_n_gram"])
            
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

    # semantic metrics
    metrics["corpus_dsi"] = corpus_dsi
    metrics["avg_dsi"] = mean([result["metrics"]["dsi"] for result in results])
    metrics["avg_surprise"] = mean([result["metrics"]["surprise"] for result in results])
    metrics["avg_n_gram_diversity"] = [mean([result["metrics"]["n_gram_diversity"][n_gram_len-1] for result in results]) for n_gram_len in range(1, config["max_n_gram"]+1)]
    
    if len(stories) > 1:
        metrics["avg_inv_homogen"] = mean([result["metrics"]["inv_homogen"] for result in results])
        metrics["avg_novelty"] = mean([result["metrics"]["novelty"] for result in results])
        metrics["avg_theme_uniqueness"] = mean([result["metrics"]["theme_uniqueness"] for result in results])

    if config["report_usage"]:
        metrics["usage"] = usage
        metrics["cost"] = cost

    return metrics

def report_metrics(results_files, config):
    result_groups = defaultdict(list)

    for results_file in results_files:
        results = read_json(results_file)
        
        try:
            if "data" in results:
                for sample in results["data"]:
                    sample["metadata"] = {
                        "source": results_file,
                        "model": results["metadata"]["model"],
                        "model_args": results["metadata"].get("model_args")
                    }

                    model_path = results["metadata"].get("model_path")
                    tokenizer_path = results["metadata"].get("tokenizer_path")

                    if model_path:
                        sample["metadata"]["model_path"] = model_path
                    if tokenizer_path:
                        sample["metadata"]["tokenizer_path"] = tokenizer_path
    
                    result_groups[sample["id"]].append(sample)
        except Exception as e:
            print(results_file)
            raise e
    
    output_dir = pathlib.Path(config["output_dir"]) if config["output_dir"] else pathlib.Path(results_files[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_id, group_results in result_groups.items():
        num_samples = config["num_samples"]

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
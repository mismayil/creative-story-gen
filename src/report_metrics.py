import argparse
from tqdm import tqdm
import pathlib
import spacy
from collections import Counter
from statistics import mean
import math

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score, euclidean_sim, manhattan_sim
from sklearn.cluster import AgglomerativeClustering

from utils import read_json, write_json, find_files, compute_usage

DEF_EMB_MODEL = "thenlper/gte-large"
DEF_EMB_TYPE = "sentence_embedding"
DEF_DIST_FN = "cosine"
DEF_SPACY_LANG = "en_core_web_sm"
DEF_PREPROCESSING_ARGS = {
    "lower": True,
    "remove_punct": True,
    "remove_stopwords": True,
    "lemmatize": True,
    "dominant_k": None
}

SPACY_ENGINE = None
EMB_MODEL = None
EMBEDDING_CACHE = {}
SPACY_CACHE = {}

# def cache(func, cache_dict):
#     def wrapper(*args, **kwargs):
#         if args in cache_dict:
#             return cache_dict[args]
#         result = func(*args, **kwargs)
#         cache_dict[args] = result
#         return result
#     return wrapper

def load_spacy_engine(language=DEF_SPACY_LANG):
    if SPACY_ENGINE:
        return SPACY_ENGINE
    SPACY_ENGINE = spacy.load(language)
    return SPACY_ENGINE

def load_emb_model(model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE):
    if EMB_MODEL:
        return EMB_MODEL

    output_value = "sentence_embedding"

    if "token" in emb_type:
        output_value = "token_embeddings"

    EMB_MODEL = SentenceTransformer(model, output_value=output_value)
    
    return EMB_MODEL

def get_embedding(text, model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE):
    # if text in EMBEDDING_CACHE:
    #     return EMBEDDING_CACHE[text]

    emb_model = load_emb_model(model, emb_type)

    embeddings = emb_model.encode(text)
    
    if embeddings.ndim == 2:
        embeddings = embeddings.mean(axis=0)

    # EMBEDDING_CACHE[text] = embeddings
    return embeddings

def compute_sem_dis(emb1, emb2, distance_fn=DEF_DIST_FN):
    if distance_fn == "cosine":
        return 1 - cos_sim(emb1, emb2)
    elif distance_fn == "dot":
        return 1 - dot_score(emb1, emb2)
    elif distance_fn == "euclidean":
        return -euclidean_sim(emb1, emb2)
    elif distance_fn == "manhattan":
        return -manhattan_sim(emb1, emb2)
    else:
        raise ValueError(f"Invalid distance function: {distance_fn}")

def get_spacy_doc(text):
    if text in SPACY_CACHE:
        return SPACY_CACHE[text]
    spacy_engine = load_spacy_engine()
    SPACY_CACHE[text] = spacy_engine(text)
    return SPACY_CACHE[text]

def get_sentences(text):
    doc = get_spacy_doc(text)
    return [sent.text for sent in doc.sents]

def get_words(text, lower=True, remove_punct=True, remove_stopwords=True, lemmatize=True, dominant_k=None):
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
    
    word_freq = Counter(words)

    if dominant_k is None or dominant_k == 0 or dominant_k >= len(words):
        return [w[0] for w in word_freq.most_common()]

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
        novelty_scores.append(2 * math.abs(story_avg_sem_dis - corpus_avg_sem_dis))
    
    return novelty_scores

def compute_theme_uniqueness(stories, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct",
                             cluster_linkage="ward", cluster_distance_threshold=0.7, preprocessing_args=DEF_PREPROCESSING_ARGS):
    story_embeddings = [compute_story_embedding(story, emb_model, emb_type, emb_strategy, preprocessing_args) for story in stories]
    clustering = AgglomerativeClustering(n_clusters=None, linkage=cluster_linkage, distance_threshold=cluster_distance_threshold)
    cluster_labels = clustering.fit_predict(story_embeddings)
    cluster_freq = Counter(cluster_labels)
    return [1 / cluster_freq[cluster_labels[i]] for i in range(len(stories))]

def compute_dsi(story, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    return compute_avg_sem_dis(story, emb_model, emb_type, distance_fn, preprocessing_args)

def compute_surprise(story, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    sentences = get_sentences(story)

    sentence_avg_sem_distances = [compute_avg_sem_dis(sentence, emb_model, emb_type, distance_fn, preprocessing_args) for sentence in sentences]
    raw_surprises = [sentence_avg_sem_distances[i] - sentence_avg_sem_distances[i - 1] for i in range(1, len(sentence_avg_sem_distances))]

    return (2 / len(raw_surprises)) * sum(raw_surprises)

def compute_metrics(results, report_usage=True):
    metrics = {}

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    cost = {
        "input": 0,
        "output": 0,
        "total": 0
    }

    template = results["data"][0]["template"]
    result_map = {}
    ref_response_attr = "reference"
    model_response_attr = "model_output"

    for result in results["data"]:
        result_map[result["id"]] = result

        if model_response_attr in result:
            if report_usage:
                sample_usage, sample_cost = compute_usage(result, results["metadata"]["model"])

                if sample_usage:
                    usage["prompt_tokens"] += sample_usage["prompt_tokens"]
                    usage["completion_tokens"] += sample_usage["completion_tokens"]
                    usage["total_tokens"] += sample_usage["total_tokens"]

                if sample_cost:
                    cost["input"] += sample_cost["input"]
                    cost["output"] += sample_cost["output"]
                    cost["total"] += sample_cost["total"]

    if report_usage:
        metrics["usage"] = usage
        metrics["cost"] = cost

    metrics["num_samples"] = len(results["data"])

    return metrics

def report_metrics(results_files, report_usage=True):
    for results_file in tqdm(results_files, total=len(results_files), desc="Reporting metrics"):
        results = read_json(results_file)
        
        try:
            if "data" in results:
                metrics = compute_metrics(results, report_usage=report_usage)
                results["metrics"] = metrics
                write_json(results, results_file)
        except Exception as e:
            print(results_file)
            raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results-path", type=str, help="Path to evaluation results file in json or directory", required=True)
    parser.add_argument("-u", "--report-usage", action="store_true", help="Report usage metrics", default=True)

    args = parser.parse_args()

    files_to_process = []

    results_path = pathlib.Path(args.results_path)

    if results_path.is_file():
        files_to_process.append(args.results_path)
    else:
        files_to_process.extend(find_files(args.results_path))

    report_metrics(files_to_process, args.report_usage)

if __name__ == "__main__":
    main()
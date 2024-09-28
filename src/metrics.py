import spacy
from collections import Counter
from statistics import mean

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score, euclidean_sim, manhattan_sim
from sklearn.cluster import AgglomerativeClustering

from utils import cache

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
    words = get_words(story, lower=True, remove_punct=False, remove_stopwords=False, lemmatize=False, unique=False)
    all_n_grams = []

    for n in range(1, max_n_gram + 1):
        all_n_grams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    
    all_n_gram_freqs = [Counter(n_grams) for n_grams in all_n_grams]
    n_gram_diversity = [len(n_gram_freqs) / len(n_grams) for n_grams, n_gram_freqs in zip(all_n_grams, all_n_gram_freqs)]

    return n_gram_diversity, all_n_gram_freqs

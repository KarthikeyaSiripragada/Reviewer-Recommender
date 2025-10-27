import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import json, re
import faiss
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- New Imports for TF-IDF ---
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ---

DEVICE = "cpu"  # CPU only

# Resolve paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_MODEL_ROOT = os.path.join(_REPO_ROOT, "models") # Central models folder

DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Paths for ALL models ---
INDEX_PATH = os.path.join(_MODEL_ROOT, "papers.index")
META_PATH  = os.path.join(_MODEL_ROOT, "meta.json")

# New paths for TF-IDF models
TFIDF_VECTORIZER_PATH = os.path.join(_MODEL_ROOT, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH = os.path.join(_MODEL_ROOT, "tfidf_matrix.joblib")
# ---

RERANK_K_FIXED = 5  # always return 5 items

# --- Helper Functions (Shared) ---

def _tok_set(s: str):
    if not s:
        return set()
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return {t for t in s.split() if len(t) > 2}

def _compute_meta_score(query, candidate_meta):
    qset = _tok_set(query)
    text = candidate_meta.get("text", "")
    tset = _tok_set(text[:600])
    if not qset or not tset:
        kw_score = 0.0
    else:
        overlap = qset & tset
        kw_score = len(overlap) / max(len(qset), 1)
        kw_score = min(1.0, kw_score * 1.0)

    year_score = 0.0
    if "year" in candidate_meta:
        try:
            y = int(candidate_meta["year"])
            # clamp to [0,1]
            year_score = min(1.0, max(0.0, (y - 2000) / 25.0))
        except Exception:
            year_score = 0.0
    return 0.8 * kw_score + 0.2 * year_score

def _load_index_and_meta_or_empty(emb_dim: int):
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Ensure meta keys are integers if loaded from json
        meta = {int(k): v for k, v in meta.items()}
        return index, meta

    print(
        "[INFO] FAISS index or metadata not found.\n"
        f"Expected:\n  {INDEX_PATH}\n  {META_PATH}\n"
        "Returning an empty index. Run embed_index.py to build the index."
    )
    return faiss.IndexFlatIP(int(emb_dim)), {}

def _load_meta_or_empty():
    """Loads only the metadata file."""
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Ensure meta keys are integers
        return {int(k): v for k, v in meta.items()}
    print(f"[INFO] Metadata not found at {META_PATH}. Returning empty.")
    return {}

# --- MODE 1: Embedding (Your original function) ---
def recommend_embedding(
    query_text,
    top_k=100,
    rerank_k=None,  # ignored; we force 5
    device=DEVICE,
    embed_model_name=DEFAULT_EMBED_MODEL,
    rerank_model_name=DEFAULT_RERANK_MODEL,
    metadata_weight=0.2,
    exclude_authors=None,
    reranker_batch_size=16
):
    """Recommends based on embeddings + CrossEncoder rerank."""
    print(f"Running Embedding recommendation for top_k={top_k}")
    
    # ---- Robust coercions
    try:
        top_k = int(top_k) if top_k is not None else 50
    except Exception:
        top_k = 50
    try:
        metadata_weight = float(metadata_weight) if metadata_weight is not None else 0.2
    except Exception:
        metadata_weight = 0.2
    try:
        reranker_batch_size = int(reranker_batch_size) if reranker_batch_size is not None else 16
    except Exception:
        reranker_batch_size = 16
    RERANK = int(RERANK_K_FIXED)

    if exclude_authors is None:
        exclude_authors = []

    # Load models
    embed_model = SentenceTransformer(embed_model_name, device=device)
    try:
        emb_dim = embed_model.get_sentence_embedding_dimension()
    except Exception:
        emb_dim = int(embed_model.encode(["x"], convert_to_numpy=True, normalize_embeddings=True).shape[-1])

    reranker = CrossEncoder(rerank_model_name, device=device)

    # Load FAISS + meta (or empty)
    index, meta = _load_index_and_meta_or_empty(emb_dim)
    if not meta:
         return {"paper_results": [], "author_rank": []}

    # Encode query
    q_emb = embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    q_emb = q_emb / norms

    # Search
    ntotal = index.ntotal if hasattr(index, "ntotal") else 0
    eff_top_k = min(top_k, max(ntotal, 1))
    D, I = index.search(q_emb, eff_top_k)

    cand_indices = [i for i in I[0].tolist() if isinstance(i, (int, np.integer)) and 0 <= i < len(meta)]
    candidates = [meta[i] for i in cand_indices]

    # Exclude by author
    if exclude_authors:
        ex = set(a.strip() for a in exclude_authors if a.strip())
        candidates = [c for c in candidates if c.get("author") not in ex]

    if not candidates:
        return {"paper_results": [], "author_rank": []}

    # Rerank
    pairs = [(query_text, c.get("text", "")[:4000]) for c in candidates]
    scores = np.array(reranker.predict(pairs, batch_size=reranker_batch_size), dtype=float)

    # Normalize 0..1
    if scores.max() == scores.min():
        semantic = np.ones_like(scores)
    else:
        semantic = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Combine with metadata
    final_scores = []
    for c, sem in zip(candidates, semantic):
        meta_score = _compute_meta_score(query_text, c)
        final_scores.append((1.0 - metadata_weight) * sem + metadata_weight * meta_score)

    ranked = sorted(zip(candidates, final_scores), key=lambda x: -x[1])
    top = ranked[:RERANK]  # always 5

    # Author agg (mean)
    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand.get("author", "Unknown")].append(float(score))
    
    # Use mean, not max, to match "avg" label in Streamlit
    author_rank = sorted([(a, float(np.mean(s))) for a, s in author_scores.items()], key=lambda x: -x[1])

    return {
        "paper_results": [(c, float(s * 100)) for c, s in top],
        "author_rank": [(a, float(s * 100)) for a, s in author_rank],
    }

# --- MODE 2: TF-IDF (Keyword) Implementation ---
def recommend_tfidf(query_text, top_k=50, exclude_authors=None):
    """
    Recommends based on TF-IDF cosine similarity.
    """
    print(f"Running TF-IDF recommendation for top_k={top_k}")

    # 1. Load pre-built TF-IDF models
    if not os.path.exists(TFIDF_VECTORIZER_PATH) or not os.path.exists(TFIDF_MATRIX_PATH):
        print("TF-IDF models not found. Run the indexing script first.")
        return {"paper_results": [], "author_rank": []}
        
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    
    # 2. Load metadata
    metadata = _load_meta_or_empty()
    if not metadata:
        return {"paper_results": [], "author_rank": []}
    
    # 3. Vectorize the query text
    query_vec = vectorizer.transform([query_text])
    
    # 4. Compute cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 5. Get top_k indices
    if top_k >= len(similarities):
        top_k = len(similarities) - 1
        
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices_sorted = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    # 6. Format results
    paper_results = []
    
    if exclude_authors is None:
        exclude_authors = []
    ex = set(a.strip() for a in exclude_authors if a.strip())

    for i in top_indices_sorted:
        score = similarities[i]
        if score > 0.01: # Set a small threshold
            cand = metadata.get(i, {})
            # Exclude authors if needed
            if cand.get("author") not in ex:
                # Score is 0-1, so multiply by 100 for percentage
                paper_results.append((cand, score * 100))

    # 7. Aggregate authors (using mean)
    author_scores = defaultdict(list)
    # Use top 5 papers for author rank, matching reranker
    for cand, score in paper_results[:RERANK_K_FIXED]: 
        author = cand.get("author", "Unknown")
        if author != "Unknown":
            author_scores[author].append(score)

    author_rank = sorted(
        [(author, np.mean(scores)) for author, scores in author_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "paper_results": paper_results[:RERANK_K_FIXED], 
        "author_rank": author_rank
    }

# --- MODE 3: LDA (Stub) ---
def recommend_lda(query_text, top_k=50, **kwargs):
    """
    Recommends based on LDA topic modeling.
    (Needs gensim library)
    """
    print("LDA mode is not implemented yet.")
    # 1. Load pre-trained LDA model and dictionary (from gensim)
    # 2. Load metadata
    # 3. Convert query_text to bag-of-words using the dictionary
    # 4. Infer topic distribution for the query
    # 5. Load pre-computed topic matrix for all documents
    # 6. Find most similar documents (e.g., Hellinger distance, cosine)
    # 7. Format results and return
    raise NotImplementedError("LDA mode must be implemented")

# --- MODE 4: Doc2Vec (Stub) ---
def recommend_doc2vec(query_text, top_k=50, **kwargs):
    """
    Recommends based on Doc2Vec.
    (Needs gensim library)
    """
    print("Doc2Vec mode is not implemented yet.")
    # 1. Load pre-trained Doc2Vec model (from gensim)
    # 2. Load metadata
    # 3. Infer vector for the query_text
    # 4. Use model.dv.most_similar() to find top_k
    # 5. Format results and return
    raise NotImplementedError("Doc2Vec mode must be implemented")


# --- MAIN ROUTER FUNCTION ---
def recommend(query_text, mode="embedding", **kwargs):
    """
    Main router function to select the recommendation strategy.
    
    :param query_text: The input text (title + abstract)
    :param mode: The strategy to use:
                 'embedding' (default), 'tfidf', 'lda', 'doc2vec'
    :param kwargs: All other arguments (top_k, rerank_k, device, etc.)
    """
    
    if mode == "embedding":
        # Pass only the relevant args to the embedding function
        return recommend_embedding(
            query_text,
            top_k=kwargs.get("top_k", 50),
            rerank_k=kwargs.get("rerank_k", 5),
            device=kwargs.get("device", "cpu"),
            embed_model_name=kwargs.get("embed_model_name"),
            rerank_model_name=kwargs.get("rerank_model_name"),
            metadata_weight=kwargs.get("metadata_weight", 0.2),
            exclude_authors=kwargs.get("exclude_authors", [])
        )
        
    elif mode == "tfidf":
        return recommend_tfidf(
            query_text,
            top_k=kwargs.get("top_k", 50),
            exclude_authors=kwargs.get("exclude_authors", [])
        )
        
    elif mode == "lda":
        return recommend_lda(
            query_text,
            top_k=kwargs.get("top_k", 50),
            **kwargs
        )
        
    elif mode == "doc2vec":
        return recommend_doc2vec(
            query_text,
            top_k=kwargs.get("top_k", 50),
            **kwargs
        )
        
    else:
        raise ValueError(f"Unknown recommendation mode: {mode}")
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import json
import re
import faiss
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Paths & defaults
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_MODEL_ROOT = os.path.join(_REPO_ROOT, "models")

INDEX_PATH = os.path.join(_MODEL_ROOT, "papers.index")
META_PATH = os.path.join(_MODEL_ROOT, "meta.json")
TFIDF_VECTORIZER_PATH = os.path.join(_MODEL_ROOT, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH = os.path.join(_MODEL_ROOT, "tfidf_matrix.joblib")

DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cpu"
RERANK_K_FIXED = 5


# --- Helpers ---
def _tok_set(s):
    if not s:
        return set()
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return {t for t in s.split() if len(t) > 2}


def _compute_meta_score(query, candidate_meta):
    qset = _tok_set(query)
    text = candidate_meta.get("text", "")
    tset = _tok_set(text[:600])

    kw_score = 0.0
    if qset and tset:
        overlap = qset & tset
        kw_score = len(overlap) / max(len(qset), 1)
        kw_score = min(1.0, kw_score)

    year_score = 0.0
    if "year" in candidate_meta:
        try:
            y = int(candidate_meta["year"])
            year_score = min(1.0, max(0.0, (y - 2000) / 25.0))
        except Exception:
            year_score = 0.0

    return 0.8 * kw_score + 0.2 * year_score


def _load_index_and_meta_or_empty(emb_dim):
    """
    Load FAISS index and metadata. Return (index, meta_dict).
    Accepts meta.json as either dict or list and converts to {int: meta}.
    If index or meta missing or unreadable, returns an empty IndexFlatIP and an (possibly empty) meta dict.
    """
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta_raw = json.load(f)

            # If it's a dict: convert keys that look like ints to int keys
            if isinstance(meta_raw, dict):
                meta = {}
                for k, v in meta_raw.items():
                    try:
                        ik = int(k)
                        meta[ik] = v
                    except Exception:
                        # skip keys that can't be converted to int
                        continue

            # If it's a list: enumerate into integer-keyed dict
            elif isinstance(meta_raw, list):
                meta = {int(i): v for i, v in enumerate(meta_raw)}

            else:
                print(f"[WARN] Unexpected meta.json type: {type(meta_raw)}. Using empty meta.")
        except Exception as e:
            print(f"[WARN] Failed reading meta.json: {e}. Using empty meta.")

    # Determine embedding dimension fallback
    try:
        emb_dim_i = int(emb_dim)
    except Exception:
        emb_dim_i = 768

    # Load FAISS index if present
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            return index, meta
        except Exception as e:
            print(f"[WARN] Could not read FAISS index at {INDEX_PATH}: {e}. Creating empty index.")

    # Fallback empty index
    print(f"[INFO] FAISS index not found or unreadable. Creating empty IndexFlatIP(dim={emb_dim_i}).")
    return faiss.IndexFlatIP(emb_dim_i), meta


def _load_meta_or_empty():
    """
    Return meta dict keyed by integer index. Accepts both dict and list JSON formats.
    """
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta_raw = json.load(f)

            if isinstance(meta_raw, dict):
                meta = {}
                for k, v in meta_raw.items():
                    try:
                        meta[int(k)] = v
                    except Exception:
                        continue
                return meta
            elif isinstance(meta_raw, list):
                return {int(i): v for i, v in enumerate(meta_raw)}
            else:
                print(f"[WARN] Unexpected meta.json type: {type(meta_raw)}. Returning empty.")
                return {}
        except Exception as e:
            print(f"[WARN] Failed reading meta.json: {e}. Returning empty.")
            return {}
    print(f"[INFO] Metadata not found at {META_PATH}. Returning empty.")
    return {}


# --- Embedding mode ---
def recommend_embedding(
    query_text,
    top_k=100,
    rerank_k=None,
    device=DEVICE,
    embed_model_name=None,
    rerank_model_name=None,
    metadata_weight=0.2,
    exclude_authors=None,
    reranker_batch_size=16,
):
    # defensive coercions
    top_k = int(top_k) if top_k is not None else 50
    metadata_weight = float(metadata_weight) if metadata_weight is not None else 0.2
    reranker_batch_size = int(reranker_batch_size) if reranker_batch_size is not None else 16
    embed_model_name = embed_model_name or DEFAULT_EMBED_MODEL
    rerank_model_name = rerank_model_name or DEFAULT_RERANK_MODEL
    exclude_authors = exclude_authors or []
    RERANK = int(RERANK_K_FIXED) if rerank_k is None else int(rerank_k)

    # Load encoder model
    embed_model = SentenceTransformer(embed_model_name, device=device)
    # determine embedding dim robustly
    emb_dim_val = None
    try:
        emb_dim_val = embed_model.get_sentence_embedding_dimension()
    except Exception:
        emb_dim_val = None

    if emb_dim_val is not None:
        try:
            emb_dim = int(emb_dim_val)
        except Exception:
            emb_dim = int(embed_model.encode(["x"], convert_to_numpy=True).shape[-1])
    else:
        emb_dim = int(embed_model.encode(["x"], convert_to_numpy=True).shape[-1])

    # Load index and meta
    index, meta = _load_index_and_meta_or_empty(emb_dim)

    # If no metadata, nothing to return
    if not meta:
        return {"paper_results": [], "author_rank": []}

    # Encode query
    q_emb = embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    # safe normalize
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    q_emb = q_emb / norms

    # Determine effective top_k based on index size
    ntotal = getattr(index, "ntotal", 0)
    if ntotal == 0:
        ntotal = max(len(meta), 0)
    eff_top_k = min(top_k, max(int(ntotal), 1))
    eff_top_k = int(eff_top_k) if eff_top_k is not None else 1

    # Run FAISS search (safe)
    q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
    try:
        distances, indices = index.search(q_emb, eff_top_k)  # type: ignore
    except Exception as e:
        print(f"[WARN] FAISS search failed: {e}")
        return {"paper_results": [], "author_rank": []}

    # Build candidate list from valid indices present in meta
    cand_indices = []
    if indices is not None and len(indices) > 0:
        for val in indices[0].tolist():
            try:
                if isinstance(val, (int, np.integer)) and (val in meta):
                    cand_indices.append(int(val))
            except Exception:
                continue

    candidates = [meta[i] for i in cand_indices]

    # Exclude authors if requested
    if exclude_authors:
        ex = {a.strip() for a in exclude_authors if a.strip()}
        candidates = [c for c in candidates if c.get("author") not in ex]

    if not candidates:
        return {"paper_results": [], "author_rank": []}

    # Rerank with CrossEncoder (fallback to embedding similarity if reranker fails)
    try:
        reranker = CrossEncoder(rerank_model_name, device=device)
        pairs = [(query_text, c.get("text", "")[:4000]) for c in candidates]
        scores = np.array(reranker.predict(pairs, batch_size=reranker_batch_size), dtype=float)
    except Exception as e:
        print(f"[WARN] Reranker failed: {e}. Falling back to embedding cosine similarity.")
        cand_texts = [c.get("text", "")[:512] for c in candidates]
        cand_embs = embed_model.encode(cand_texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        # cosine similarity manual (cand_embs dot q_emb)
        q_vec = q_emb[0]
        denom = (np.linalg.norm(cand_embs, axis=1) * (np.linalg.norm(q_vec) + 1e-9))
        scores = np.dot(cand_embs, q_vec) / (denom + 1e-9)
        scores = np.array(scores, dtype=float)

    # Normalize semantic scores to 0..1
    if scores.size == 0:
        return {"paper_results": [], "author_rank": []}
    semantic = np.ones_like(scores) if scores.max() == scores.min() else (
        (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    )

    # Combine semantic and metadata score
    final_scores = []
    for c, sem in zip(candidates, semantic):
        meta_score = _compute_meta_score(query_text, c)
        final_scores.append((1.0 - metadata_weight) * float(sem) + metadata_weight * float(meta_score))

    ranked = sorted(zip(candidates, final_scores), key=lambda x: -x[1])
    top = ranked[:RERANK]

    # Aggregate author scores (mean)
    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand.get("author", "Unknown")].append(float(score))

    author_rank = sorted([(a, float(np.mean(s))) for a, s in author_scores.items()], key=lambda x: -x[1])

    return {
        "paper_results": [(c, float(s * 100)) for c, s in top],
        "author_rank": [(a, float(s * 100)) for a, s in author_rank],
    }


# --- TF-IDF mode ---
def recommend_tfidf(query_text, top_k=50, exclude_authors=None):
    if not os.path.exists(TFIDF_VECTORIZER_PATH) or not os.path.exists(TFIDF_MATRIX_PATH):
        print("TF-IDF models not found. Run the indexing script first.")
        return {"paper_results": [], "author_rank": []}

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    metadata = _load_meta_or_empty()
    if not metadata:
        return {"paper_results": [], "author_rank": []}

    query_vec = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if similarities.size == 0:
        return {"paper_results": [], "author_rank": []}

    top_k = int(min(max(1, int(top_k or 50)), similarities.size))
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices_sorted = top_indices[np.argsort(similarities[top_indices])[::-1]]

    ex = {a.strip() for a in (exclude_authors or []) if a.strip()}
    paper_results = []
    for i in top_indices_sorted:
        score = float(similarities[int(i)])
        if score > 0.01:
            cand = metadata.get(int(i), {})
            if cand.get("author") not in ex:
                paper_results.append((cand, score * 100.0))

    # author agg
    author_scores = defaultdict(list)
    for cand, score in paper_results[:RERANK_K_FIXED]:
        author = cand.get("author", "Unknown")
        if author != "Unknown":
            author_scores[author].append(score)

    author_rank = sorted([(a, float(np.mean(s))) for a, s in author_scores.items()], key=lambda x: -x[1])
    return {"paper_results": paper_results[:RERANK_K_FIXED], "author_rank": author_rank}


# --- Stubs ---
def recommend_lda(query_text, top_k=50, **kwargs):
    raise NotImplementedError("LDA mode must be implemented")


def recommend_doc2vec(query_text, top_k=50, **kwargs):
    raise NotImplementedError("Doc2Vec mode must be implemented")


# --- Router ---
def recommend(query_text, mode="embedding", **kwargs):
    mode = mode or "embedding"
    if mode == "embedding":
        return recommend_embedding(
            query_text,
            top_k=kwargs.get("top_k", 50),
            rerank_k=kwargs.get("rerank_k", 5),
            device=kwargs.get("device", "cpu"),
            embed_model_name=kwargs.get("embed_model_name") or DEFAULT_EMBED_MODEL,
            rerank_model_name=kwargs.get("rerank_model_name") or DEFAULT_RERANK_MODEL,
            metadata_weight=kwargs.get("metadata_weight", 0.2),
            exclude_authors=kwargs.get("exclude_authors", []),
            reranker_batch_size=kwargs.get("reranker_batch_size", 16),
        )
    elif mode == "tfidf":
        return recommend_tfidf(
            query_text,
            top_k=kwargs.get("top_k", 50),
            exclude_authors=kwargs.get("exclude_authors", []),
        )
    elif mode == "lda":
        top_k_val = int(kwargs.pop("top_k", 50))
        return recommend_lda(query_text, top_k=top_k_val, **kwargs)
    elif mode == "doc2vec":
        top_k_val = int(kwargs.pop("top_k", 50))
        return recommend_doc2vec(query_text, top_k=top_k_val, **kwargs)
    else:
        raise ValueError(f"Unknown recommendation mode: {mode}")

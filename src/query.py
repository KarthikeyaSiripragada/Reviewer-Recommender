# src/query.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import torch
import re
# ---- Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Resolve paths relative to repo root (src/..)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

# ---- Default model names
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Data paths
INDEX_PATH = os.path.join(_REPO_ROOT, "models", "papers.index")
META_PATH  = os.path.join(_REPO_ROOT, "models", "meta.json")

# ──────────────────────────────────────────────────────────────────────────────
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
            year_score = min(1.0, max(0.0, (y - 2000) / 25.0))
        except Exception:
            year_score = 0.0

    return 0.8 * kw_score + 0.2 * year_score

def _load_index_and_meta_or_empty(emb_dim: int):
    """Try to load FAISS + meta; if missing, return an empty index + empty meta."""
    idx_exists = os.path.exists(INDEX_PATH)
    meta_exists = os.path.exists(META_PATH)
    if idx_exists and meta_exists:
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta

    # Fallback to empty to keep UI alive
    print(
        "[INFO] FAISS index or metadata not found.\n"
        f"Expected:\n  {INDEX_PATH}\n  {META_PATH}\n"
        "Returning an empty index. Run embed_index.py to build the index."
    )
    return faiss.IndexFlatIP(emb_dim), []

# ──────────────────────────────────────────────────────────────────────────────
def recommend(
    query_text,
    top_k=100,
    rerank_k=10,
    device=DEVICE,
    embed_model_name=DEFAULT_EMBED_MODEL,
    rerank_model_name=DEFAULT_RERANK_MODEL,
    metadata_weight=0.2,
    exclude_authors=None,
    reranker_batch_size=16
):
    """
    Returns:
      {
        "paper_results": [(candidate_meta, final_score_percent), ...],
        "author_rank":   [(author, avg_score_percent), ...]
      }
    """
    if exclude_authors is None:
        exclude_authors = []

    # Load models first (to know embedding dim for empty-index fallback)
    embed_model = SentenceTransformer(embed_model_name, device=device)
    try:
        emb_dim = embed_model.get_sentence_embedding_dimension()
    except Exception:
        emb_dim = int(embed_model.encode(["x"], convert_to_numpy=True, normalize_embeddings=True).shape[-1])

    reranker = CrossEncoder(rerank_model_name, device=device)

    # Load FAISS + meta (or empty)
    index, meta = _load_index_and_meta_or_empty(emb_dim)

    # Embed query
    q_emb = embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True); norms[norms == 0] = 1e-9
    q_emb = q_emb / norms

    # Search
    ntotal = index.ntotal if hasattr(index, "ntotal") else 0
    eff_top_k = min(int(top_k), max(ntotal, 1))  # at least 1 to satisfy faiss
    D, I = index.search(q_emb, eff_top_k)

    cand_indices = [i for i in I[0].tolist() if isinstance(i, (int, np.integer)) and 0 <= i < len(meta)]
    candidates = [meta[i] for i in cand_indices]

    # Exclude by author
    if exclude_authors:
        ex = set(a.strip() for a in exclude_authors if a.strip())
        candidates = [c for c in candidates if c.get("author") not in ex]

    if not candidates:
        print("[WARN] No candidates found (empty index or all excluded).")
        return {"paper_results": [], "author_rank": []}

    # Rerank
    pairs = [(query_text, c.get("text", "")[:4000]) for c in candidates]
    scores = np.array(reranker.predict(pairs, batch_size=reranker_batch_size), dtype=float)

    # Normalize 0..1
    if scores.max() == scores.min():
        semantic = np.ones_like(scores)
    else:
        semantic = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Combine with metadata score
    final_scores = []
    for c, sem in zip(candidates, semantic):
        meta_score = _compute_meta_score(query_text, c)
        final_scores.append((1.0 - metadata_weight) * sem + metadata_weight * meta_score)

    ranked = sorted(zip(candidates, final_scores), key=lambda x: -x[1])
    top = ranked[: int(rerank_k)]

    # Aggregate by author (max)
    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand.get("author", "Unknown")].append(float(score))
    author_rank = sorted([(a, float(max(s))) for a, s in author_scores.items()], key=lambda x: -x[1])

    # Console summary
    print("\n🔍 Query:", query_text)
    print("\n🏆 Top Recommended Reviewers:\n")
    for i, (cand, score) in enumerate(top, 1):
        pct = round(float(score) * 100, 2)
        snippet = cand.get("text", "")[:120].replace("\n", " ")
        print(f"{i}. Author: {cand.get('author','Unknown')}")
        print(f"   Final score: {pct}% (semantic + metadata combined)")
        print(f"   Paper: {cand.get('file', 'Unknown')}")
        print(f"   Snippet: {snippet}...\n")

    print("📊 Author Ranking Summary:")
    for i, (author, avg_score) in enumerate(author_rank, 1):
        print(f"{i}. {author} — Score: {round(avg_score * 100, 2)}%")

    return {
        "paper_results": [(c, float(s * 100)) for c, s in top],
        "author_rank": [(a, float(s * 100)) for a, s in author_rank],
    }

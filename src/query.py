# src/query.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import torch
import os
import re

# Auto GPU detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default model names (can be overridden when calling recommend)
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_PATH = "models/papers.index"
META_PATH = "models/meta.json"

def load_resources_index():
    """Load FAISS index and metadata only (fast)."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Missing index or meta files. Run embed_index.py first.")
    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    return index, meta

# simple tokenizer for keyword overlap
def _tok_set(s):
    if not s: return set()
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return set([t for t in s.split() if len(t) > 2])

def _compute_meta_score(query, candidate_meta):
    """
    Compute optional meta score in [0,1] using:
      - keyword overlap between query and candidate text/title
      - recency if 'year' present (prefers newer papers)
    If metadata absent, returns 0.
    """
    qset = _tok_set(query)
    text = candidate_meta.get("text","")
    tset = _tok_set(text[:600])  # focus on title+abstract region
    if not qset or not tset:
        kw_score = 0.0
    else:
        overlap = qset & tset
        kw_score = len(overlap) / max(len(qset), 1)
        kw_score = min(1.0, kw_score * 1.0)  # scale

    year_score = 0.0
    if "year" in candidate_meta:
        try:
            # normalize years between 0..1 using a simple heuristic: newer -> higher
            y = int(candidate_meta["year"])
            year_score = min(1.0, max(0.0, (y - 2000) / 25.0))
        except Exception:
            year_score = 0.0

    # weighted sum
    return 0.8 * kw_score + 0.2 * year_score

def recommend(
    query_text,
    top_k=100,
    rerank_k=10,
    device=DEVICE,
    embed_model_name=DEFAULT_EMBED_MODEL,
    rerank_model_name=DEFAULT_RERANK_MODEL,
    metadata_weight=0.2,              # weight in [0,1] used to combine meta score
    exclude_authors=None,             # list of author names to exclude (conflict-of-interest)
    reranker_batch_size=16
):
    """
    Retrieve top_k papers from FAISS, rerank with a CrossEncoder, and aggregate by author.
    Returns structure:
      {"paper_results": [(candidate_meta, final_score_percent), ...],
       "author_rank": [(author, avg_score_percent), ...]}
    Notes:
      - final_score = (1 - metadata_weight) * semantic + metadata_weight * meta_score
      - semantic score is normalized reranker score (0..1)
    """
    if exclude_authors is None:
        exclude_authors = []

    index, meta = load_resources_index()

    # Load models dynamically
    embed_model = SentenceTransformer(embed_model_name, device=device)
    reranker = CrossEncoder(rerank_model_name, device=device)

    # Step 1: Embed query and search FAISS (safe normalization)
    q_emb = embed_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    q_emb = np.asarray(q_emb, dtype=np.float32)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    q_emb = q_emb / norms
    D, I = index.search(q_emb, top_k)
    candidates = [meta[i] for i in I[0] if i < len(meta)]

    # Exclude candidates by author (conflict-of-interest)
    if exclude_authors:
        candidates = [c for c in candidates if c.get("author") not in set(exclude_authors)]

    if not candidates:
        print("[WARN] No candidates found from FAISS (or all excluded).")
        return {"paper_results": [], "author_rank": []}

    # Step 2: Rerank top candidates with CrossEncoder
    pairs = [(query_text, c.get("text","")[:4000]) for c in candidates]
    scores = reranker.predict(pairs, batch_size=reranker_batch_size)
    scores = np.array(scores, dtype=float)

    # Normalize reranker scores to 0..1
    if scores.max() == scores.min():
        semantic = np.ones_like(scores) * 1.0
    else:
        semantic = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Step 3: Combine with metadata score (if any)
    final_scores = []
    for c, sem in zip(candidates, semantic):
        meta_score = _compute_meta_score(query_text, c)  # 0..1
        final = (1.0 - metadata_weight) * sem + metadata_weight * meta_score
        final_scores.append(final)

    # Sort and pick top rerank_k
    ranked = sorted(zip(candidates, final_scores), key=lambda x: -x[1])
    top = ranked[:rerank_k]

    # Step 4: Aggregate to author level using max (favor best paper)
    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand.get("author","Unknown")].append(float(score))
    author_rank = sorted(
        [(a, float(max(s))) for a, s in author_scores.items()],
        key=lambda x: -x[1]
    )

    # Step 5: Print clean results (percentage)
    print("\n🔍 Query:", query_text)
    print("\n🏆 Top Recommended Reviewers:\n")
    for i, (cand, score) in enumerate(top, 1):
        pct = round(float(score) * 100, 2)
        snippet = cand.get("text","")[:120].replace("\n", " ")
        print(f"{i}. Author: {cand.get('author','Unknown')}")
        print(f"   Final score: {pct}% (semantic + metadata combined)")
        print(f"   Paper: {cand.get('file', 'Unknown')}")
        print(f"   Snippet: {snippet}...\n")

    print("📊 Author Ranking Summary:")
    for i, (author, avg_score) in enumerate(author_rank, 1):
        print(f"{i}. {author} — Score: {round(avg_score*100, 2)}%")

    # Return percent scores for UI convenience
    paper_results = [(c, float(s*100)) for c, s in top]
    author_rank_pct = [(a, float(s*100)) for a, s in author_rank]
    return {"paper_results": paper_results, "author_rank": author_rank_pct}
# src/query.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import torch
import os
import re

# Auto GPU detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default model names (can be overridden when calling recommend)
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_PATH = "models/papers.index"
META_PATH = "models/meta.json"

def load_resources_index():
    """Load FAISS index and metadata only (fast)."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Missing index or meta files. Run embed_index.py first.")
    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    return index, meta

# simple tokenizer for keyword overlap
def _tok_set(s):
    if not s: return set()
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return set([t for t in s.split() if len(t) > 2])

def _compute_meta_score(query, candidate_meta):
    """
    Compute optional meta score in [0,1] using:
      - keyword overlap between query and candidate text/title
      - recency if 'year' present (prefers newer papers)
    If metadata absent, returns 0.
    """
    qset = _tok_set(query)
    text = candidate_meta.get("text","")
    tset = _tok_set(text[:600])  # focus on title+abstract region
    if not qset or not tset:
        kw_score = 0.0
    else:
        overlap = qset & tset
        kw_score = len(overlap) / max(len(qset), 1)
        kw_score = min(1.0, kw_score * 1.0)  # scale

    year_score = 0.0
    if "year" in candidate_meta:
        try:
            # normalize years between 0..1 using a simple heuristic: newer -> higher
            y = int(candidate_meta["year"])
            year_score = min(1.0, max(0.0, (y - 2000) / 25.0))
        except Exception:
            year_score = 0.0

    # weighted sum
    return 0.8 * kw_score + 0.2 * year_score

def recommend(
    query_text,
    top_k=100,
    rerank_k=10,
    device=DEVICE,
    embed_model_name=DEFAULT_EMBED_MODEL,
    rerank_model_name=DEFAULT_RERANK_MODEL,
    metadata_weight=0.2,              # weight in [0,1] used to combine meta score
    exclude_authors=None,             # list of author names to exclude (conflict-of-interest)
    reranker_batch_size=16
):
    """
    Retrieve top_k papers from FAISS, rerank with a CrossEncoder, and aggregate by author.
    Returns structure:
      {"paper_results": [(candidate_meta, final_score_percent), ...],
       "author_rank": [(author, avg_score_percent), ...]}
    Notes:
      - final_score = (1 - metadata_weight) * semantic + metadata_weight * meta_score
      - semantic score is normalized reranker score (0..1)
    """
    if exclude_authors is None:
        exclude_authors = []

    index, meta = load_resources_index()

    # Load models dynamically
    embed_model = SentenceTransformer(embed_model_name, device=device)
    reranker = CrossEncoder(rerank_model_name, device=device)

    # Step 1: Embed query and search FAISS (safe normalization)
    q_emb = embed_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    q_emb = np.asarray(q_emb, dtype=np.float32)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    q_emb = q_emb / norms
    D, I = index.search(q_emb, top_k)
    candidates = [meta[i] for i in I[0] if i < len(meta)]

    # Exclude candidates by author (conflict-of-interest)
    if exclude_authors:
        candidates = [c for c in candidates if c.get("author") not in set(exclude_authors)]

    if not candidates:
        print("[WARN] No candidates found from FAISS (or all excluded).")
        return {"paper_results": [], "author_rank": []}

    # Step 2: Rerank top candidates with CrossEncoder
    pairs = [(query_text, c.get("text","")[:4000]) for c in candidates]
    scores = reranker.predict(pairs, batch_size=reranker_batch_size)
    scores = np.array(scores, dtype=float)

    # Normalize reranker scores to 0..1
    if scores.max() == scores.min():
        semantic = np.ones_like(scores) * 1.0
    else:
        semantic = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Step 3: Combine with metadata score (if any)
    final_scores = []
    for c, sem in zip(candidates, semantic):
        meta_score = _compute_meta_score(query_text, c)  # 0..1
        final = (1.0 - metadata_weight) * sem + metadata_weight * meta_score
        final_scores.append(final)

    # Sort and pick top rerank_k
    ranked = sorted(zip(candidates, final_scores), key=lambda x: -x[1])
    top = ranked[:rerank_k]

    # Step 4: Aggregate to author level using max (favor best paper)
    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand.get("author","Unknown")].append(float(score))
    author_rank = sorted(
        [(a, float(max(s))) for a, s in author_scores.items()],
        key=lambda x: -x[1]
    )

    # Step 5: Print clean results (percentage)
    print("\n🔍 Query:", query_text)
    print("\n🏆 Top Recommended Reviewers:\n")
    for i, (cand, score) in enumerate(top, 1):
        pct = round(float(score) * 100, 2)
        snippet = cand.get("text","")[:120].replace("\n", " ")
        print(f"{i}. Author: {cand.get('author','Unknown')}")
        print(f"   Final score: {pct}% (semantic + metadata combined)")
        print(f"   Paper: {cand.get('file', 'Unknown')}")
        print(f"   Snippet: {snippet}...\n")

    print("📊 Author Ranking Summary:")
    for i, (author, avg_score) in enumerate(author_rank, 1):
        print(f"{i}. {author} — Score: {round(avg_score*100, 2)}%")

    # Return percent scores for UI convenience
    paper_results = [(c, float(s*100)) for c, s in top]
    author_rank_pct = [(a, float(s*100)) for a, s in author_rank]
    return {"paper_results": paper_results, "author_rank": author_rank_pct}

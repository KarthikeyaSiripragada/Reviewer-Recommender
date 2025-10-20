# src/query.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import torch
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

def recommend(
    query_text,
    top_k=100,
    rerank_k=10,
    device=DEVICE,
    embed_model_name=DEFAULT_EMBED_MODEL,
    rerank_model_name=DEFAULT_RERANK_MODEL
):
    """
    Retrieve top_k papers from FAISS, rerank with a CrossEncoder, and aggregate by author.
    Parameters:
      - embed_model_name: sentence-transformers model used to embed query (e.g., all-mpnet-base-v2)
        Short: MPNet gives strong semantic embeddings (best balance accuracy/speed).
      - rerank_model_name: cross-encoder model to score (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
        Short: CrossEncoder attends to query+candidate jointly for high precision.
    """
    print(f"Loading resources on {device} (embed={embed_model_name}, rerank={rerank_model_name}) ...")
    index, meta = load_resources_index()


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

    if not candidates:
        print("[WARN] No candidates found from FAISS.")
        return {"paper_results": [], "author_rank": []}


    pairs = [(query_text, c["text"][:4000]) for c in candidates]
    scores = reranker.predict(pairs, batch_size=16)


    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        norm_scores = np.ones_like(scores) * 100.0
    else:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) * 100.0

    ranked = sorted(zip(candidates, norm_scores), key=lambda x: -x[1])
    top = ranked[:rerank_k]

    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand["author"]].append(float(score))
    author_rank = sorted([(a, float(max(s))) for a,s in author_scores.items()], key=lambda x:-x[1])


    print("\n🔍 Query:", query_text)
    print("\n🏆 Top Recommended Reviewers:\n")
    for i, (cand, score) in enumerate(top, 1):
        snippet = cand["text"][:120].replace("\n", " ")
        print(f"{i}. Author: {cand.get('author','Unknown')}")
        print(f"   Similarity: {round(score, 2)}%")
        print(f"   Paper: {cand.get('file', 'Unknown')}")
        print(f"   Snippet: {snippet}...\n")

    print("📊 Author Ranking Summary:")
    for i, (author, avg_score) in enumerate(author_rank, 1):
        print(f"{i}. {author} — Avg Similarity: {round(avg_score, 2)}%")

    return {"paper_results": top, "author_rank": author_rank}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Neural networks for image classification"
    recommend(q, top_k=50, rerank_k=10, device=DEVICE)

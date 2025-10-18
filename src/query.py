# src/query.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import torch
import os

# Auto GPU detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model names
EMBED_MODEL = "all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
INDEX_PATH = "models/papers.index"
META_PATH = "models/meta.json"

def load_resources(device=DEVICE):
    """Load FAISS index, metadata, embedding & rerank models."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Missing index or meta files. Run embed_index.py first.")

    print(f"Loading models on {device} ...")
    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    reranker = CrossEncoder(RERANK_MODEL, device=device)
    return index, meta, embed_model, reranker

def recommend(query_text, top_k=100, rerank_k=10, device=DEVICE):
    """Retrieve top_k papers from FAISS, rerank with CrossEncoder, and aggregate by author."""
    index, meta, embed_model, reranker = load_resources(device=device)

    # Step 1: Embed query and search FAISS
    q_emb = embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    candidates = [meta[i] for i in I[0] if i < len(meta)]

    # Step 2: Rerank top candidates with CrossEncoder
    pairs = [(query_text, c["text"][:4000]) for c in candidates]
    scores = reranker.predict(pairs, batch_size=16)

    # Step 3: Normalize reranker scores to 0–100 scale
    scores = np.array(scores)
    if scores.max() == scores.min():
        norm_scores = np.ones_like(scores) * 100
    else:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) * 100

    ranked = sorted(zip(candidates, norm_scores), key=lambda x: -x[1])
    top = ranked[:rerank_k]

    # Step 4: Aggregate to author level
    author_scores = defaultdict(list)
    for cand, score in top:
        author_scores[cand["author"]].append(float(score))
    author_rank = sorted(
        [(a, float(np.mean(s))) for a, s in author_scores.items()],
        key=lambda x: -x[1]
    )

    # Step 5: Print clean results
    print("\n🔍 Query:", query_text)
    print("\n🏆 Top Recommended Reviewers:\n")
    for i, (cand, score) in enumerate(top, 1):
        snippet = cand["text"][:100].replace("\n", " ")
        print(f"{i}. Author: {cand['author']}")
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

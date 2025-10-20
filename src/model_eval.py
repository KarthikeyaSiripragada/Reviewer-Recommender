import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.query import recommend
EMBED_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-mpnet-base-dot-v1"
]


RERANK_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2"
]


QUERIES = [
    "deep learning for medical image classification",
    "optimization of neural networks for low-power devices",
    "natural language processing for sentiment analysis"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Running model evaluation on {DEVICE.upper()}...\n")

def safe_recommend(query, embed_model_name, rerank_model_name):
    try:
        start = time.time()
        res = recommend(
            query,
            top_k=50,
            rerank_k=10,
            device=DEVICE,
            embed_model_name=embed_model_name,
            rerank_model_name=rerank_model_name
        )
        duration = time.time() - start
        mean_sim = np.mean([r["score"] for r in res["paper_results"]])
        return mean_sim, duration
    except Exception as e:
        print(f"⚠️ {embed_model_name} + {rerank_model_name} failed: {e}")
        return None, None

records = []
for em in EMBED_MODELS:
    for rm in RERANK_MODELS:
        print(f"🔹 Evaluating {em} + {rm}")
        sim_scores, times = [], []
        for q in QUERIES:
            sim, t = safe_recommend(q, em, rm)
            if sim is not None:
                sim_scores.append(sim)
                times.append(t)
        if sim_scores:
            records.append({
                "Embedding": em,
                "Reranker": rm,
                "MeanSimilarity": round(np.mean(sim_scores), 3),
                "AvgRuntime(s)": round(np.mean(times), 2)
            })
        print("──────────────────────────")
if records:
    df = pd.DataFrame(records)
    df = df.sort_values(by="MeanSimilarity", ascending=False)
    df.to_csv("model_eval_results.csv", index=False)
    print("\n✅ Evaluation complete — saved to model_eval_results.csv\n")
    print(df.to_string(index=False))
else:
    print("❌ No successful results. Check model names or internet connection.")

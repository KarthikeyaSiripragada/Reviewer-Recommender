import traceback
from src.query import recommend

def run_test():
    try:
        # run on CPU to avoid CUDA errors; set small top_k/rerank_k
        recommend(
    "test query about neural networks",
    top_k=10,
    rerank_k=5,
    device="cpu",
    embed_model_name="all-mpnet-base-v2",  # ✅ same as embed_index.py
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
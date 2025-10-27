# src/embed_index.py
import os, json, numpy as np, re, faiss
from sentence_transformers import SentenceTransformer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

DATA_DIR = os.path.join(_REPO_ROOT, "data", "authors")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

MODELPATH = "all-mpnet-base-v2"
INDEX_PATH = os.path.join(_REPO_ROOT, "models", "papers.index")
META_PATH  = os.path.join(_REPO_ROOT, "models", "meta.json")
EMB_PATH   = os.path.join(_REPO_ROOT, "models", "embeddings.npy")

DEVICE = "cpu"  # CPU-only

def clean_text(text: str) -> str:
    text = text.replace("-\n", "")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"Page\s*\d+", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def load_texts_from_authors(data_dir=DATA_DIR):
    items = []
    if not os.path.isdir(data_dir):
        print(f"[ERR] Data dir not found: {data_dir}")
        return items
    for author in os.listdir(data_dir):
        ap = os.path.join(data_dir, author)
        if not os.path.isdir(ap):
            continue
        for fname in os.listdir(ap):
            if not fname.lower().endswith((".pdf", ".txt")):
                continue
            txt_path = os.path.join(PROCESSED_DIR, f"{os.path.splitext(fname)[0]}.txt")
            if not os.path.exists(txt_path):
                print(f"[WARN] No processed txt for {author}/{fname}; skipping.")
                continue
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = clean_text(f.read())[:1200]
            if text.strip():
                items.append({"id": len(items), "author": author, "file": fname, "text": text})
    return items

def build_index(items, model_name=MODELPATH, device: str = DEVICE):
    print(f"[INFO] Building FAISS index using model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    texts = [it["text"] for it in items]
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    os.makedirs(os.path.join(_REPO_ROOT, "models"), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved index → {INDEX_PATH}")
    print(f"[OK] Saved meta  → {META_PATH}")
    print(f"[OK] Saved embeddings → {EMB_PATH}")

if __name__ == "__main__":
    items = load_texts_from_authors()
    if not items:
        print("[ERR] No items found. Make sure processed text files exist.")
    else:
        build_index(items)

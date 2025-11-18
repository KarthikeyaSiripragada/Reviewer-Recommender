# src/embed_index.py — safer + correct FAISS index builder

import os
import json
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
from tempfile import NamedTemporaryFile
from typing import cast

# -----------------------------
# Paths
# -----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

DATA_DIR = os.path.join(_REPO_ROOT, "data", "authors")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

MODELPATH = "all-mpnet-base-v2"
INDEX_PATH = os.path.join(_REPO_ROOT, "models", "papers.index")
META_PATH = os.path.join(_REPO_ROOT, "models", "meta.json")
EMB_PATH = os.path.join(_REPO_ROOT, "models", "embeddings.npy")

DEVICE = "cpu"


# -----------------------------
# Utilities
# -----------------------------
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

            txt_path = os.path.join(
                PROCESSED_DIR, f"{os.path.splitext(fname)[0]}.txt"
            )

            if not os.path.exists(txt_path):
                print(f"[WARN] No processed txt for {author}/{fname}; skipping.")
                continue

            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = clean_text(f.read())[:1200]
            except Exception as e:
                print(f"[ERR] Could not read file {txt_path}: {e}")
                continue

            if text.strip():
                items.append({
                    "id": len(items),
                    "author": author,
                    "file": fname,
                    "text": text
                })

    return items


def _atomic_write(path: str, data_bytes: bytes):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    with NamedTemporaryFile(delete=False, dir=folder) as tmp:
        tmp.write(data_bytes)
        tmp.flush()
        tmp_name = tmp.name

    os.replace(tmp_name, path)


# -----------------------------
# Index building
# -----------------------------
def build_index(items, model_name: str = MODELPATH, device: str = DEVICE):
    """
    Build FAISS index over processed items.
    items: list of dicts {id, author, file, text}
    """
    print(f"[INFO] Building FAISS index using model: {model_name} on {device}")

    model = SentenceTransformer(model_name, device=device)

    # Encode texts
    texts = [it["text"] for it in items]
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS index (Inner Product)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)

    # Convert embeddings to contiguous float32
    arr = np.ascontiguousarray(embs.astype(np.float32))

    # Add vectors to the index
    index.add(arr) # pyright: ignore[reportCallIssue]

    # Save all model files atomically
    try:
        os.makedirs(os.path.join(_REPO_ROOT, "models"), exist_ok=True)

        # Save FAISS index
        tmp_index = INDEX_PATH + ".tmp"
        faiss.write_index(index, tmp_index)
        os.replace(tmp_index, INDEX_PATH)

        # Save embeddings
        np.save(EMB_PATH + ".tmp", embs)
        os.replace(EMB_PATH + ".tmp.npy", EMB_PATH)

        # Save meta.json (list)
        meta_bytes = json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write(META_PATH, meta_bytes)

    except Exception as e:
        print(f"[ERR] Failed saving model artifacts: {e}")
        raise

    print(f"[OK] Saved index → {INDEX_PATH}")
    print(f"[OK] Saved meta  → {META_PATH}")
    print(f"[OK] Saved embeddings → {EMB_PATH}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    items = load_texts_from_authors()

    if not items:
        print("[ERR] No items found. Make sure processed text files exist.")
    else:
        build_index(items)

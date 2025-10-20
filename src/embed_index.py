# src/embed_index.py
import os
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss

# ─────────────────────────────────────────────
DATA_DIR = "data/authors"
PROCESSED_DIR = "data/processed"
MODELPATH = "all-mpnet-base-v2"   # embedding model (change to MiniLM if desired)
INDEX_PATH = "models/papers.index"
META_PATH = "models/meta.json"
EMB_PATH = "models/embeddings.npy"
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic cleanup: remove hyphenation, headers, page numbers, multi-newlines."""
    text = text.replace("-\n", "")                       # fix hyphenated line breaks
    text = re.sub(r"\n+", " ", text)                     # flatten newlines
    text = re.sub(r"Page\s*\d+", "", text, flags=re.I)   # remove 'Page X'
    text = re.sub(r"\s{2,}", " ", text)                  # collapse spaces
    return text.strip()

def load_texts_from_authors(data_dir=DATA_DIR):
    """Loads text from authors' folders, preferring processed .txt files."""
    items = []
    for author in os.listdir(data_dir):
        author_path = os.path.join(data_dir, author)
        if not os.path.isdir(author_path):
            continue

        for fname in os.listdir(author_path):
            if not fname.lower().endswith((".pdf", ".txt")):
                continue

            text = ""
            txt_path = os.path.join(PROCESSED_DIR, f"{os.path.splitext(fname)[0]}.txt")

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                    text = clean_text(raw)
                    # Keep only title + abstract length (approx.)
                    text = text[:1200]
            else:
                print(f"[WARN] No processed txt for {author}/{fname}; skipping.")
                continue

            if text.strip():
                items.append({
                    "id": len(items),
                    "author": author,
                    "file": fname,
                    "text": text
                })

    return items

def build_index(items, model_name=MODELPATH, device="cuda"):
    """Encodes all text, builds a cosine FAISS index."""
    print(f"[INFO] Building FAISS index using model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    texts = [it["text"] for it in items]
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # auto-normalize for cosine
    )

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    os.makedirs("models", exist_ok=True)
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

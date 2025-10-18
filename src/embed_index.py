# src/embed_index.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = "data/authors"       # expects: data/authors/<author_name>/*.pdf or .txt
PROCESSED_DIR = "data/processed" # optional: if you saved text files
MODELPATH = "all-mpnet-base-v2"  # high-quality; uses GPU if device='cuda'
INDEX_PATH = "models/papers.index"
META_PATH = "models/meta.json"
EMB_PATH = "models/embeddings.npy"

def load_texts_from_authors(data_dir=DATA_DIR):
    items = []
    for author in os.listdir(data_dir):
        a_path = os.path.join(data_dir, author)
        if not os.path.isdir(a_path): continue
        for fname in os.listdir(a_path):
            if not fname.lower().endswith((".pdf", ".txt")): continue
            fpath = os.path.join(a_path, fname)
            # if .pdf, try reading preprocessed .txt in processed dir, else try raw .txt
            text = ""
            if fname.lower().endswith(".txt"):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                # if you already extracted pdfs to data/processed, check there
                txtf = os.path.join(PROCESSED_DIR, f"{os.path.splitext(fname)[0]}.txt")
                if os.path.exists(txtf):
                    with open(txtf, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                else:
                    # fallback: skip PDF here (preferred to preprocess first)
                    print(f"[WARN] No processed txt for {fpath}; skip or preprocess first.")
                    continue
            items.append({
                "id": len(items),
                "author": author,
                "file": fname,
                "text": text
            })
    return items

def build_index(items, model_name=MODELPATH, device="cuda"):
    model = SentenceTransformer(model_name, device=device)
    texts = [it["text"] for it in items]
    # compute embeddings (batch on GPU)
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine with IndexFlatIP
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved index ({INDEX_PATH}), meta ({META_PATH}), embeddings ({EMB_PATH})")

if __name__ == "__main__":
    items = load_texts_from_authors()
    if not items:
        print("[ERR] No items found. Make sure you have data/processed/<paper>.txt or author txts.")
    else:
        build_index(items)

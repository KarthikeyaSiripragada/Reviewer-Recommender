import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ.setdefault("USE_TIKA", "0")
APP_VERSION = "v2025-10-27-1619"

# path fix
import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# --- end of path fix ---

# Now this import will work
import traceback
import pandas as pd
import streamlit as st
from src.query import recommend, INDEX_PATH, META_PATH # Combined imports

st.set_page_config(page_title="Reviewer Recommender", layout="wide")

# ---- Sidebar: minimal + vertical
st.sidebar.markdown("### System Status")
st.sidebar.info("Running on **CPU**")
st.sidebar.caption(f"Build: {APP_VERSION}")

# Presence badges (no object dump)
import os as _os
if _os.path.exists(INDEX_PATH):
    st.sidebar.success("Index present")
else:
    st.sidebar.error("Index MISSING")
if _os.path.exists(META_PATH):
    st.sidebar.success("Meta present")
else:
    st.sidebar.error("Meta MISSING")

EMBED_OPTIONS = {
    "all-mpnet-base-v2": "MPNet — high-quality sentence embeddings (768-dim).",
    "all-MiniLM-L6-v2": "MiniLM — fast/distilled encoder (384-dim).",
    "multi-qa-mpnet-base-dot-v1": "Multi-QA MPNet — tuned for retrieval.",
}
RERANK_OPTIONS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM CrossEncoder (6-layer).",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "MiniLM CrossEncoder (12-layer).",
}

with st.sidebar:
    st.markdown("### Settings")
    
    # --- NEW: Mode Selector ---
    recommend_mode = st.selectbox(
        "Recommendation Mode",
        ["embedding", "tfidf", "lda", "doc2vec"],
        index=0,
        help="Select the retrieval strategy. 'embedding' is the default SBERT+Reranker. 'tfidf' is keyword-based."
    )
    # ---
    
    embed_model = st.selectbox("Embedding model", list(EMBED_OPTIONS.keys()), index=0)
    rerank_model = st.selectbox("Reranker model", list(RERANK_OPTIONS.keys()), index=0)
    top_k = st.number_input("Retrieve top_k", min_value=10, max_value=1000, value=50, step=10)
    metadata_weight = st.slider("Metadata weight", 0.0, 1.0, 0.2, 0.05)
    exclude_authors_raw = st.text_input("Exclude authors (comma-separated)", value="")

st.subheader("1) Upload PDF")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
extracted_text = ""
if uploaded_file:
    os.makedirs("data/sample_input", exist_ok=True)
    save_path = os.path.join("data/sample_input", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Saved to `{save_path}`")
    try:
        with st.spinner("Extracting..."):
            from src.parse_pdf import extract_text
            extracted_text = extract_text(save_path)
        st.success("Extraction complete")
    except Exception:
        st.error("Extraction failed")
        st.text(traceback.format_exc())
else:
    st.info("Or paste text manually below to test the recommender.")

st.subheader("2) Text (title + abstract recommended)")
text_input = st.text_area("Paper text", value=extracted_text, height=260, placeholder="Paste title + abstract here...")

col1, col2 = st.columns([1, 1])
run_btn = col1.button("Run recommender")
save_btn = col2.button("Save text")

if save_btn:
    os.makedirs("data/sample_input", exist_ok=True)
    out_file = os.path.join("data/sample_input", "edited.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text_input)
    st.success(f"Saved edited text to `{out_file}`")

if run_btn:
    if not text_input or len(text_input.strip()) < 20:
        st.warning("Provide text (title + abstract recommended).")
    else:
        exclude_authors = [a.strip() for a in exclude_authors_raw.split(",") if a.strip()]
        
        # ---- SAFE coercions BEFORE calling recommend (avoid int/float on None)
        try:
            top_k_val = int(top_k) if top_k is not None else 50
        except Exception:
            top_k_val = 50
        try:
            metadata_weight_val = float(metadata_weight) if metadata_weight is not None else 0.2
        except Exception:
            metadata_weight_val = 0.2

        try:
            with st.spinner(f"Running recommendation (mode: {recommend_mode})..."):
                res = recommend(
                    text_input,
                    mode=recommend_mode,  # <-- NEW: Pass the selected mode
                    top_k=top_k_val,
                    rerank_k=5,           # Pass a default integer
                    device="cpu",
                    embed_model_name=embed_model,
                    rerank_model_name=rerank_model,
                    metadata_weight=metadata_weight_val,
                    exclude_authors=exclude_authors,
                )
            st.success("Done")

            if not res["paper_results"]:
                st.info("No results yet — build the index files locally (`python -m src.embed_index` etc.) and commit the `models/` folder.")

            # ---- Results (lean vertical)
            rows = []
            st.markdown("### Top papers (final score %)")
            for i, (cand, score) in enumerate(res.get("paper_results", []), start=1):
                st.markdown(f"**{i}. {cand.get('author','Unknown')}** — {round(float(score),2)}%")
                st.caption(cand.get("file", "Unknown"))
                snippet = (cand.get("text","")[:400] or "").replace("\n", " ")
                st.text_area(f"Snippet {i}", snippet + " ...", height=110, key=f"snip{i}")
                rows.append({
                    "rank": i,
                    "author": cand.get("author","Unknown"),
                    "file": cand.get("file","Unknown"),
                    "score": float(score)
                })

            st.markdown("### Author ranking")
            rows_auth = []
            for i, (author, avg) in enumerate(res.get("author_rank", []), start=1):
                st.write(f"{i}. **{author}** — {round(float(avg),2)}%")
                rows_auth.append({"rank": i, "author": author, "score": float(avg)})

            if rows:
                df = pd.DataFrame(rows)
                st.download_button("Download paper results CSV", df.to_csv(index=False).encode("utf-8"),
                                    file_name="paper_results.csv", mime="text/csv")
            if rows_auth:
                df2 = pd.DataFrame(rows_auth)
                st.download_button("Download author ranking CSV", df2.to_csv(index=False).encode("utf-8"),
                                    file_name="author_rank.csv", mime="text/csv")
        
        except NotImplementedError as e:
            st.error(f"This mode is not implemented yet: {e}")
        except Exception as e:
            st.error("An error occurred during recommendation:")
            st.text(traceback.format_exc())


st.markdown("---")
st.markdown(
    "**Project Submission:** This application was developed by "
    "**Karthikeya Siripragada (SE22UECM018)** and "
    "**Karthik Raj Gupta (SE22UCAM004)** as part of project coursework."
)
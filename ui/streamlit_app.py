# ui/streamlit_app.py
import os
# Avoid TF/Keras/JAX pull-ins
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ.setdefault("USE_TIKA", "0")  # default: no Java/Tika
APP_VERSION = "v2025-10-27-1330"

import traceback
import torch
import pandas as pd
import streamlit as st
from src.query import INDEX_PATH, META_PATH
import os
st.sidebar.write("Index path:", INDEX_PATH)
st.sidebar.write("Meta path:", META_PATH)
st.sidebar.success("Index present") if os.path.exists(INDEX_PATH) else st.sidebar.error("Index MISSING")
st.sidebar.success("Meta present")  if os.path.exists(META_PATH)  else st.sidebar.error("Meta MISSING")

# Ensure repo root on sys.path
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.parse_pdf import extract_text
from src.query import recommend

# Fixed CPU-only
device = "cpu"

st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.info("Running on **CPU**")

EMBED_OPTIONS = {
    "all-mpnet-base-v2": "MPNet — high-quality sentence embeddings (768-dim).",
    "all-MiniLM-L6-v2": "MiniLM — fast/distilled encoder (384-dim).",
    "multi-qa-mpnet-base-dot-v1": "Multi-QA MPNet — tuned for retrieval."
}
RERANK_OPTIONS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM CrossEncoder (6-layer).",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "MiniLM CrossEncoder (12-layer)."
}

with st.sidebar:
    st.header("Settings")
    embed_model = st.selectbox("Embedding model", list(EMBED_OPTIONS.keys()), index=0)
    rerank_model = st.selectbox("Reranker model", list(RERANK_OPTIONS.keys()), index=0)
    top_k = st.number_input("FAISS top_k (retrieve)", min_value=10, max_value=1000, value=50, step=10)
    rerank_k = st.number_input("Rerank top_k (final)", min_value=1, max_value=50, value=10, step=1)
    metadata_weight = st.slider("Metadata weight (0=semantic only)", 0.0, 1.0, 0.2, 0.05)
    exclude_authors_raw = st.text_input("Exclude authors (comma-separated)", value="")
    st.markdown("---")
    st.write("Embedding info:")
    st.write(EMBED_OPTIONS[embed_model])
    st.write("Reranker info:")
    st.write(RERANK_OPTIONS[rerank_model])

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
            extracted_text = extract_text(save_path)
        st.success("Extraction complete")
    except Exception:
        st.error("Extraction failed")
        st.text(traceback.format_exc())
else:
    st.info("Or paste text manually below to test the recommender.")

st.subheader("2) Text (title+abstract recommended)")
text_input = st.text_area("Paper text", value=extracted_text, height=300)

col1, col2 = st.columns([1,1])
run_btn = col1.button("Run recommender")
save_btn = col2.button("Save text")

if save_btn:
    os.makedirs("data/sample_input", exist_ok=True)
    out_file = os.path.join("data/sample_input", "edited.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text_input)
    st.success(f"Saved edited text to `{out_file}`")

# run and show
if run_btn:
    if not text_input or len(text_input.strip()) < 20:
        st.warning("Provide text (title + abstract recommended).")
    else:
        exclude_authors = [a.strip() for a in exclude_authors_raw.split(",") if a.strip()]
        with st.spinner("Running recommendation..."):
            res = recommend(
                text_input,
                top_k=int(top_k),
                rerank_k=int(rerank_k),
                device="cpu",
                embed_model_name=embed_model,
                rerank_model_name=rerank_model,
                metadata_weight=float(metadata_weight),
                exclude_authors=exclude_authors
            )
        st.success("Done")

        if not res["paper_results"]:
            st.info(
                "No results yet — build the FAISS index locally with "
                "`python -m src.embed_index` and commit the `models/` folder."
            )

        # prepare dataframe for download
        rows = []
        st.subheader("Top papers (final score %)")
        for i, (cand, score) in enumerate(res.get("paper_results", []), start=1):
            st.markdown(f"**{i}. {cand.get('author','Unknown')}** — {round(float(score),2)}%")
            st.write(f"**File:** {cand.get('file','Unknown')}")
            snippet = cand.get("text","")[:400].replace("\n", " ")
            st.text_area(f"Snippet {i}", snippet + " ...", height=120, key=f"snip{i}")
            rows.append({"rank": i, "author": cand.get("author","Unknown"), "file": cand.get("file","Unknown"), "score": float(score)})

        st.subheader("Author ranking")
        rows_auth = []
        for i, (author, avg) in enumerate(res.get("author_rank", []), start=1):
            st.write(f"{i}. **{author}** — Score: {round(float(avg),2)}%")
            rows_auth.append({"rank": i, "author": author, "score": float(avg)})

        if rows:
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download paper results CSV", csv, file_name="paper_results.csv", mime="text/csv")

        if rows_auth:
            df2 = pd.DataFrame(rows_auth)
            csv2 = df2.to_csv(index=False).encode('utf-8')
            st.download_button("Download author ranking CSV", csv2, file_name="author_rank.csv", mime="text/csv")
st.sidebar.caption(f"Build: {APP_VERSION}")
st.markdown(
    "**Project Submission:** This application was developed by "
    "**Karthikeya Siripragada (SE22UECM018)** and "
    "**Karthik Raj Gupta (SE22UCAM004)** as part of project coursework."
)

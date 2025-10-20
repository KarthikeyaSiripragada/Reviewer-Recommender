import os
import streamlit as st
import traceback
from src.parse_pdf import extract_text
from src.query import recommend
import torch

st.set_page_config(page_title="Reviewer Recommender", page_icon="🧠", layout="centered")

st.title("Reviewer Recommender")
st.markdown(
    "Upload a paper (PDF) → extract text → run the reviewer recommender on the extracted text."
)
EMBED_OPTIONS = {
    "all-mpnet-base-v2": "MPNet — high-quality sentence embeddings (768-dim). Good accuracy, moderate speed.",
    "all-MiniLM-L6-v2": "MiniLM — distilled lightweight encoder (384-dim). Fast and cheap, slightly less precise.",
    "multi-qa-mpnet-base-dot-v1": "Multi-QA MPNet — tuned for question↔passage retrieval (dot-product). Great for semantic search."
}
RERANK_OPTIONS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM CrossEncoder (6-layer) — fast reranker, good precision.",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "MiniLM CrossEncoder (12-layer) — deeper, higher precision but slower."
}

DEVICE_AUTO = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.header("Settings")
    embed_model = st.selectbox("Embedding model", list(EMBED_OPTIONS.keys()), index=0)
    rerank_model = st.selectbox("Reranker model", list(RERANK_OPTIONS.keys()), index=0)
    st.checkbox("Use GPU (if available)", value=(DEVICE_AUTO == "cuda"), key="use_gpu")
    top_k = st.number_input("FAISS top_k (retrieve)", min_value=10, max_value=1000, value=50, step=10)
    rerank_k = st.number_input("Rerank top_k (final)", min_value=1, max_value=50, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model quick info**")
with st.sidebar.expander("Embedding model info", expanded=False):
    st.write(EMBED_OPTIONS[embed_model])
with st.sidebar.expander("Reranker model info", expanded=False):
    st.write(RERANK_OPTIONS[rerank_model])

st.subheader("1) Upload PDF to extract text")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

extracted_text = ""
if uploaded_file:
    os.makedirs("data/sample_input", exist_ok=True)
    save_path = os.path.join("data/sample_input", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Saved to `{save_path}`")
    st.info("Extracting text — this may take a few seconds.")
    try:
        with st.spinner("Extracting..."):
            extracted_text = extract_text(save_path)
        st.success("Extraction complete")
    except Exception as e:
        st.error("Extraction failed — check server logs")
        st.text(traceback.format_exc())

if not uploaded_file:
    st.info("Or paste text manually below to test the recommender.")

st.subheader("2) Preview / Edit text")
text_input = st.text_area(
    "Paper text (title + abstract recommended)",
    value=extracted_text,
    height=300,
    max_chars=200000
)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Run recommender on this text")
with col2:
    save_btn = st.button("Save this text to data/sample_input/edited.txt")

if save_btn:
    os.makedirs("data/sample_input", exist_ok=True)
    out_file = os.path.join("data/sample_input", "edited.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text_input)
    st.success(f"Saved edited text to `{out_file}`")
if run_btn:
    if not text_input or len(text_input.strip()) < 20:
        st.warning("Please provide some text (title + abstract recommended).")
    else:
        device = "cuda" if st.session_state.get("use_gpu", False) and torch.cuda.is_available() else "cpu"
        st.info(f"Running recommend (embed={embed_model}, rerank={rerank_model}) on {device.upper()}")

        try:
            with st.spinner("Computing recommendations..."):
                results = recommend(
                    text_input,
                    top_k=int(top_k),
                    rerank_k=int(rerank_k),
                    device=device,
                    embed_model_name=embed_model,
                    rerank_model_name=rerank_model
                )
            # results structure: {"paper_results": [(candidate, score), ...], "author_rank": [(author, avg_score), ...]}
            st.success("Recommendations ready")
            st.subheader("Top papers (reranked):")
            for i, (cand, score) in enumerate(results.get("paper_results", []), start=1):
                st.markdown(f"**{i}. {cand.get('author','Unknown')}** — {round(float(score),2)}%")
                st.write(f"**File:** {cand.get('file','Unknown')}")
                snippet = cand.get("text", "")[:400].replace("\n", " ")
                st.text_area(f"Snippet {i}", snippet + " ...", height=120, key=f"snip{i}")

            st.subheader("Author ranking:")
            for i, (author, avg) in enumerate(results.get("author_rank", []), start=1):
                st.write(f"{i}. **{author}** — Avg similarity: {round(float(avg),2)}%")

        except Exception as e:
            st.error("Recommendation failed — see error below.")
            st.text(traceback.format_exc())

st.markdown("---")
st.caption("Tip: prefer feeding title + abstract text for best results.")

# ui/streamlit_app.py
import streamlit as st
from src.parse_pdf import extract_text
import os

st.set_page_config(page_title="Reviewer Recommender", page_icon="🧠")

st.title("🧾 Paper Text Extractor (Step 1 Demo)")
st.caption("Upload a PDF paper and view the extracted text")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Save temporarily
    os.makedirs("data/sample_input", exist_ok=True)
    temp_path = os.path.join("data/sample_input", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Extracting text… this may take a few seconds.")
    text = extract_text(temp_path)

    st.success("✅ Extraction complete!")
    st.subheader("Extracted Text Preview:")
    st.text_area("Paper Text", text[:3000], height=400)

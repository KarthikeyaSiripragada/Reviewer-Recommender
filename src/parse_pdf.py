# src/parse_pdf.py
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
from tika import parser
import os

def extract_text(pdf_path: str) -> str:
    """
    Extracts raw text content from a PDF using Apache Tika.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    raw = parser.from_file(pdf_path)
    text = raw.get("content", "")
    if not text:
        print(f"[WARN] No text extracted from {pdf_path}")
    return text.strip()

def batch_extract(input_dir: str, output_dir: str):
    """
    Loops through all PDFs in a folder and saves extracted text to /processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                txt_path = os.path.join(output_dir, file.replace(".pdf", ".txt"))
                try:
                    text = extract_text(pdf_path)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"[OK] Parsed {file}")
                except Exception as e:
                    print(f"[ERR] Failed {file}: {e}")

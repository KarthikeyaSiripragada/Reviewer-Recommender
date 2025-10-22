# src/parse_pdf.py
import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Try Tika first (best for complex PDFs). Fallback to PyPDF2 if Tika/Java not available.
try:
    from tika import parser
    TIKA_OK = True
except Exception:
    TIKA_OK = False

# PyPDF2 fallback
try:
    import PyPDF2
    PYPDF2_OK = True
except Exception:
    PYPDF2_OK = False

def _extract_with_tika(pdf_path: str) -> str:
    raw = parser.from_file(pdf_path)
    return raw.get("content", "") or ""

def _extract_with_pypdf2(pdf_path: str) -> str:
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(text)

def extract_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if TIKA_OK:
        try:
            txt = _extract_with_tika(pdf_path)
            if txt and len(txt.strip()) > 20:
                return txt.strip()
        except Exception:
            pass

    if PYPDF2_OK:
        try:
            txt = _extract_with_pypdf2(pdf_path)
            if txt and len(txt.strip()) > 0:
                return txt.strip()
        except Exception:
            pass

    # Last resort: return empty string
    return ""

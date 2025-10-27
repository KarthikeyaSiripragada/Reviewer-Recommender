# src/parse_pdf.py
import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import shutil

def _java_available() -> bool:
    return shutil.which("java") is not None

USE_TIKA = os.environ.get("USE_TIKA", "0") in {"1", "true", "True", "YES", "yes"}

TIKA_OK = False
if USE_TIKA and _java_available():
    try:
        from tika import parser
        TIKA_OK = True
    except Exception:
        TIKA_OK = False

try:
    import PyPDF2
    PYPDF2_OK = True
except Exception:
    PYPDF2_OK = False

def _extract_with_tika(pdf_path: str) -> str:
    raw = parser.from_file(pdf_path)
    return (raw.get("content") or "").strip()

def _extract_with_pypdf2(pdf_path: str) -> str:
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                text.append(t)
            except Exception:
                continue
    return "\n".join(text).strip()

def extract_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if TIKA_OK:
        try:
            txt = _extract_with_tika(pdf_path)
            if len(txt) > 20:
                return txt
        except Exception:
            pass

    if PYPDF2_OK:
        try:
            txt = _extract_with_pypdf2(pdf_path)
            if txt:
                return txt
        except Exception:
            pass

    return ""

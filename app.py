# app.py — safe Streamlit entry (import UI from ui/streamlit_app.py)
import sys
from pathlib import Path
import importlib

# ensure repo root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# import UI module that defines the Streamlit app
streamlit_app = importlib.import_module("ui.streamlit_app")

if __name__ == "__main__":
    try:
        from streamlit.web import cli as stcli
    except Exception:
        import streamlit.cli as stcli  # type: ignore
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

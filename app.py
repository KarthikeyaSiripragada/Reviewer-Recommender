# app.py — safe Streamlit launcher (no double-runtime issues)

import sys
import os
import subprocess
from pathlib import Path

# --- Repo root path ---
REPO_ROOT = str(Path(__file__).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# -----------------------------------------------------------------------------------
# Find streamlit_app.py cleanly
# -----------------------------------------------------------------------------------
def find_target():
    candidates = [
        Path(REPO_ROOT) / "src" / "ui" / "streamlit_app.py",
        Path(REPO_ROOT) / "ui" / "streamlit_app.py",
        Path(REPO_ROOT) / "src" / "streamlit_app.py",
        Path(REPO_ROOT) / "streamlit_app.py",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # fallback
    return str(Path(REPO_ROOT) / "src" / "ui" / "streamlit_app.py")


# -----------------------------------------------------------------------------------
# Detect if Streamlit already spawned this interpreter
# -----------------------------------------------------------------------------------
def running_under_streamlit_env():
    for k in os.environ.keys():
        if "STREAMLIT" in k.upper():
            return True
    return False


# -----------------------------------------------------------------------------------
# Launch Streamlit as a subprocess
# -----------------------------------------------------------------------------------
def run_streamlit_subprocess(target_path: str):
    python = sys.executable or "python"
    cmd = [python, "-m", "streamlit", "run", target_path]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[app.py] Streamlit subprocess failed: {e}")
        raise


# -----------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    target = find_target()

    # If Streamlit already started us → do nothing, let Streamlit import the UI itself
    if running_under_streamlit_env():
        sys.exit(0)

    # Normal: `python app.py` → spawn Streamlit
    run_streamlit_subprocess(target)

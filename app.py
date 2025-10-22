# app.py (at repo root)
import sys
from streamlit.web import cli as stcli

# run your existing UI file
sys.argv = ["streamlit", "run", "ui/streamlit_app.py", "--server.port=7860"]
sys.exit(stcli.main())

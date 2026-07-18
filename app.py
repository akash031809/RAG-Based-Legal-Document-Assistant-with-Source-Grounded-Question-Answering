# Root-level entry point for Streamlit Cloud deployment
# Streamlit Cloud looks for app.py or a configured main file at the repo root
# This simply re-runs the actual frontend app

import runpy
import sys
from pathlib import Path

# Add root to sys.path
root = Path(__file__).parent
sys.path.insert(0, str(root))

# Run the actual frontend app
runpy.run_path(str(root / "frontend" / "app.py"), run_name="__main__")

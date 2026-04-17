"""
app.py — India Credit Risk Intelligence
JPMorgan-style institutional dashboard entrypoint.

Run: streamlit run app.py
"""
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "streamlit", "run",
     "src/visualization/build_dashboard.py"] + sys.argv[1:]
)
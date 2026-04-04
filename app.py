"""
app.py — India Credit Risk Intelligence
Single entrypoint for the dashboard.

Run:
    streamlit run app.py
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "streamlit", "run",
                "src/visualization/build_dashboard.py"] + sys.argv[1:])
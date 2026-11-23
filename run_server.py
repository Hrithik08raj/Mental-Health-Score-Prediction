"""
MindScore Server Launcher
Launches the FastAPI backend server using Uvicorn.
"""

import sys
import webbrowser
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please ensure 'Mental_Health_Model.pkl' is located in the project directory.")
        sys.exit(1)

    print("=" * 65)
    print("   🧠 MindScore – Mental Health Score Prediction System")
    print("   FastAPI High-Performance Machine Learning Backend")
    print("=" * 65)
    print("  * Web Application URL : http://127.0.0.1:8000")
    print("  * Interactive API Docs : http://127.0.0.1:8000/docs")
    print("  * ReDoc API Reference  : http://127.0.0.1:8000/redoc")
    print("  * Health Check Endpoint: http://127.0.0.1:8000/api/health")
    print("=" * 65)
    print("Starting Uvicorn server on http://127.0.0.1:8000 ... (Press CTRL+C to stop)\n")

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# One-command local setup for MedAssist RAG.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment (venv/)..."
python -m venv venv

echo "==> Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example..."
  cp .env.example .env
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "WARNING: 'ollama' not found on PATH. Install it from https://ollama.com/download"
else
  echo "==> Ensuring llama3 model is pulled..."
  ollama pull llama3
fi

echo "==> Running ingestion pipeline (downloads ~100 FDA drug labels)..."
python scripts/ingest.py

echo "==> Setup complete."
echo "Start the API:      uvicorn api.main:app --reload"
echo "Start the frontend:  streamlit run frontend/app.py"
echo "Or run everything via Docker: docker-compose up --build"

# MedAssist RAG

**Medical Document Q&A System using Retrieval-Augmented Generation (RAG)** over real FDA drug label data — built entirely with free, locally-runnable components.

> ⚠️ **Disclaimer:** This is a portfolio/engineering project, not a medical device. Answers are generated from ingested FDA drug label excerpts only and are **not** a substitute for professional medical advice.

## Architecture

```
                        ┌─────────────────┐
                        │   openFDA API    │
                        │  (drug labels)   │
                        └────────┬─────────┘
                                 │  fda_downloader.py
                                 ▼
                     ┌───────────────────────┐
                     │   data/raw/*.json      │
                     └────────────┬───────────┘
                                  │  chunker.py (semantic, section-aware)
                                  ▼
                     ┌───────────────────────┐
                     │  data/processed/       │
                     │  chunks.jsonl           │
                     └────────────┬───────────┘
                                  │  embeddings.py (all-MiniLM-L6-v2, local)
                                  ▼
                     ┌───────────────────────┐
                     │      ChromaDB           │◄──────────┐
                     │  (persistent vector db) │            │
                     └────────────┬───────────┘            │
                                  │ similarity_search        │ add_documents
                                  ▼                          │
   ┌─────────────┐    ┌───────────────────────┐    ┌────────┴────────┐
   │  Streamlit   │◄──►│      FastAPI            │    │  scripts/ingest.py│
   │  Frontend    │    │  /query /health          │    └──────────────────┘
   │  (chat UI)   │    │  /sources /stats         │
   └─────────────┘    └────────────┬───────────┘
                                  │  rag/chain.py
                                  ▼
                     ┌───────────────────────┐
                     │   LLM Provider          │
                     │  Ollama (default, local)│
                     │  Groq / OpenAI / Claude  │
                     │  (optional, cloud)       │
                     └───────────────────────┘
```

## Features

- **Fully local by default** — no API keys required to run end-to-end (Ollama + local embeddings + local ChromaDB).
- **Section-aware semantic chunking** — clinical sections (indications, warnings, dosage, etc.) are chunked as coherent units, not blindly split by character count.
- **Source-cited answers** — every answer lists the drug name and label section it came from, plus a DailyMed link.
- **Multi-provider LLM support** — swap between Ollama, Groq, OpenAI, or Anthropic per-request via a provider factory.
- **Structured API** — FastAPI with Pydantic schemas, health checks, and collection stats.
- **Chat-style frontend** — Streamlit UI with source expanders, latency/confidence display, and a live system-status sidebar.
- **RAGAS evaluation** — automated answer relevancy / faithfulness / context precision scoring, plus a provider comparison (latency, cost, accuracy) benchmark.
- **One-command Docker deployment** — `docker-compose up --build` runs the API, frontend, and ChromaDB together.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (Llama3), optionally Groq / OpenAI / Anthropic |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), local |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| Backend | FastAPI + Pydantic |
| Frontend | Streamlit |
| Data source | [openFDA drug label API](https://open.fda.gov/apis/drug/label/) |
| Evaluation | RAGAS |
| Deployment | Docker Compose |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) with the `llama3` model pulled: `ollama pull llama3`
- Docker Desktop (only needed for the containerized deployment)

### Local (no Docker)

```bash
# 1. Clone and enter the project
cd medassist-rag

# 2. One-command setup (venv, deps, .env, model pull, ingestion)
bash scripts/setup.sh

# 3. Start the API
uvicorn api.main:app --reload

# 4. In another terminal, start the frontend
streamlit run frontend/app.py
```

Then open http://localhost:8501.

### Manual setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3
python scripts/ingest.py --limit 100
uvicorn api.main:app --reload
```

### Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

This starts:
- `chromadb` on `localhost:8200`
- `api` (FastAPI) on `localhost:8000`
- `frontend` (Streamlit) on `localhost:8501`

Ollama itself runs on the **host** (not containerized) — the `api` service reaches it via `host.docker.internal:11434`. Run `python scripts/ingest.py` once locally (or exec into the `api` container) to populate the vector store before querying.

## API Documentation

Interactive docs at `http://localhost:8000/docs` once the API is running. Summary:

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Ask a question. Body: `{question, top_k?, provider?}` → `{answer, sources, confidence, response_time_ms, model_used}` |
| `GET` | `/health` | API/vector-db/model health check |
| `GET` | `/sources` | List all ingested drug names |
| `GET` | `/stats` | Vector store collection stats |

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the warnings for ibuprofen?", "top_k": 5, "provider": "ollama"}'
```

## Running Tests

```bash
pytest tests/ -v
```

Ingestion and RAG-chain logic tests run fully offline (the LLM and vector store are mocked). API tests use FastAPI's `TestClient` with a mocked chain/store, so the whole suite runs without Ollama, ChromaDB, or Docker.

## Benchmarking

```bash
# RAGAS evaluation against 20 auto-generated Q&A pairs
python benchmarks/evaluate.py

# Compare all configured providers on accuracy / latency / cost
python benchmarks/benchmark_providers.py
```

Results (JSON + a comparison chart PNG) are saved to `benchmarks/results/`.

### Sample Results

From a smoke run against 705 ingested chunks (100 openFDA drug labels), CPU-only inference, `llama3` as both the answer generator and the RAGAS judge:

| Provider | Avg Latency (generation) | Faithfulness | Answer Relevancy | Context Precision | Est. Cost/Query |
|---|---|---|---|---|---|
| Ollama (llama3, CPU) | ~12s (n=2 smoke test; single real queries took 55-115s) | 0.50 | 0.00 | 0.50 | $0.00 |

Two things stand out and are worth calling out rather than hiding:
- **CPU latency is the main cost of "fully free."** A single `/query` call against `llama3` on CPU took 55-115 seconds in testing. A GPU (or a cloud provider like Groq) brings this down to low single-digit seconds — see `rag/llm_provider.py` for how to switch.
- **`answer_relevancy` of 0.0 on a 2-question smoke test reflects the judge model, not the pipeline.** RAGAS's `answer_relevancy` metric asks the LLM to reverse-generate questions from its own answer and compares them by embedding similarity; an 8B local model is noticeably weaker at this than a frontier model. Re-run `benchmarks/evaluate.py` with a larger `n_questions` and/or a stronger `judge_provider` (e.g. `groq`) for a more representative score.

Run `python benchmarks/benchmark_providers.py` yourself to regenerate this table (and a `provider_comparison.png` chart) against your own ingested data.

## Screenshots

*(Add screenshots of the Streamlit chat UI and API docs here.)*

## Future Improvements

- Streaming responses (token-by-token) in the Streamlit UI.
- Hybrid search (BM25 + vector) for better recall on drug-name-exact queries.
- Reranking retrieved chunks with a cross-encoder before generation.
- Conversation memory / multi-turn follow-up questions.
- Authentication and per-user rate limiting on the API.
- CI pipeline running `pytest` and a lint check on every PR.

## Project Structure

```
medassist-rag/
├── ingestion/       # openFDA download, PDF extraction, semantic chunking
├── vectorstore/      # embeddings + ChromaDB wrapper
├── rag/              # LLM provider factory, prompts, RAG chain
├── api/               # FastAPI app, routes, Pydantic schemas
├── frontend/          # Streamlit chat UI
├── benchmarks/        # RAGAS evaluation + provider comparison
├── scripts/            # setup.sh, ingest.py
└── tests/              # pytest suite (ingestion, RAG chain, API)
```

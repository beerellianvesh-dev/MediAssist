"""API endpoints for MedAssist RAG: /query, /health, /sources, /stats."""

import logging

from fastapi import APIRouter, HTTPException, Request

from api.schemas import HealthResponse, QueryRequest, QueryResponse, SourcesResponse, StatsResponse
from rag.llm_provider import LLMProviderError, check_ollama_available

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer a medical question using retrieval-augmented generation."""
    chain = request.app.state.rag_chain
    try:
        result = chain.query(question=body.question, top_k=body.top_k, provider=body.provider)
    except LLMProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Failed to reach the '{body.provider}' LLM provider. "
                "If using Ollama, make sure it is running (`ollama serve`) and the model is pulled."
            ),
        ) from exc

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence_score,
        response_time_ms=result.response_time_ms,
        model_used=result.model_used,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report vector DB reachability and Ollama availability."""
    store = request.app.state.chroma_store
    try:
        count = store.get_collection_stats()["document_count"]
        db_status = "ok"
    except Exception:
        logger.exception("Vector DB health check failed")
        count = 0
        db_status = "unreachable"

    model_status = "available" if check_ollama_available() else "unavailable"

    overall = "ok" if db_status == "ok" else "degraded"
    return HealthResponse(status=overall, vector_db_count=count, model_status=model_status)


@router.get("/sources", response_model=SourcesResponse)
def sources(request: Request) -> SourcesResponse:
    """List all drugs currently ingested into the vector store."""
    store = request.app.state.chroma_store
    result = store.list_sources()
    return SourcesResponse(drugs=result["drugs"], total_documents=result["total_documents"])


@router.get("/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    """Return vector store collection statistics."""
    store = request.app.state.chroma_store
    result = store.get_collection_stats()
    return StatsResponse(collection_name=result["collection_name"], document_count=result["document_count"])

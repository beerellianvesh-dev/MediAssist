"""API tests. The vector store and RAG chain are mocked at startup so these
tests run without ChromaDB, sentence-transformers, or Ollama needing to be live."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.chain import RAGResponse


@pytest.fixture
def client():
    mock_store = MagicMock()
    mock_store.get_collection_stats.return_value = {
        "collection_name": "medassist_drug_labels",
        "document_count": 42,
    }
    mock_store.list_sources.return_value = {"drugs": ["Advil", "Tylenol"], "total_documents": 42}

    mock_chain = MagicMock()
    mock_chain.query.return_value = RAGResponse(
        answer="Advil may cause stomach bleeding. [Advil - Warnings]",
        sources=[{"drug_name": "Advil", "section_name": "Warnings", "source_url": "https://example.com"}],
        confidence_score=0.82,
        response_time_ms=123.4,
        model_used="ollama",
    )

    with patch("api.main.ChromaStore", return_value=mock_store), \
         patch("api.main.RAGChain", return_value=mock_chain), \
         patch("rag.llm_provider.check_ollama_available", return_value=True):
        from api.main import app

        with TestClient(app) as test_client:
            yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "MedAssist RAG API"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_db_count"] == 42


def test_stats(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["document_count"] == 42


def test_sources(client):
    response = client.get("/sources")
    assert response.status_code == 200
    body = response.json()
    assert "Advil" in body["drugs"]
    assert body["total_documents"] == 42


def test_query_success(client):
    response = client.post("/query", json={"question": "What are Advil's side effects?"})
    assert response.status_code == 200
    body = response.json()
    assert "stomach bleeding" in body["answer"]
    assert body["confidence"] == 0.82
    assert body["model_used"] == "ollama"
    assert len(body["sources"]) == 1


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422

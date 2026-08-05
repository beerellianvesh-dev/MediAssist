"""Tests for the RAG chain and prompt formatting. LLM calls are mocked so these
tests do not require Ollama (or any other provider) to be running."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.chain import RAGChain
from rag.prompts import format_context

SAMPLE_HITS = [
    {
        "id": "advil_1",
        "text": "Advil may cause stomach bleeding in some patients.",
        "metadata": {
            "drug_name": "Advil",
            "section_name": "Warnings",
            "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc",
        },
        "distance": 0.2,
    },
    {
        "id": "advil_2",
        "text": "Take Advil with food to reduce stomach upset.",
        "metadata": {
            "drug_name": "Advil",
            "section_name": "Dosage and Administration",
            "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc",
        },
        "distance": 0.35,
    },
]


def test_format_context_includes_drug_and_section():
    context = format_context(SAMPLE_HITS)
    assert "Advil" in context
    assert "Warnings" in context
    assert "stomach bleeding" in context


def test_confidence_from_hits_averages_similarity():
    chain = RAGChain.__new__(RAGChain)  # bypass __init__ (no real store needed)
    confidence = chain._confidence_from_hits(SAMPLE_HITS)
    # similarities: 1-0.2=0.8, 1-0.35=0.65 -> avg 0.725
    assert abs(confidence - 0.725) < 1e-6


def test_confidence_from_hits_empty_is_zero():
    chain = RAGChain.__new__(RAGChain)
    assert chain._confidence_from_hits([]) == 0.0


def test_query_returns_no_context_message_when_no_hits():
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = []
    chain = RAGChain(store=mock_store)

    response = chain.query("What is the dosage for a drug not in the database?")

    assert "don't have enough information" in response.answer
    assert response.sources == []
    assert response.confidence_score == 0.0


def test_query_builds_answer_and_sources_from_llm():
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = SAMPLE_HITS
    chain = RAGChain(store=mock_store)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Advil may cause stomach bleeding. [Advil - Warnings]")

    with patch("rag.chain.get_llm", return_value=mock_llm):
        response = chain.query("What are Advil's side effects?")

    assert "stomach bleeding" in response.answer
    assert len(response.sources) == 2
    assert response.sources[0]["drug_name"] == "Advil"
    assert response.model_used == "ollama"
    assert response.response_time_ms >= 0

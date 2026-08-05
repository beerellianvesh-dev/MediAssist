"""RAG chain: retrieve relevant drug label chunks, then generate a cited answer."""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from rag.llm_provider import get_llm
from rag.prompts import QA_PROMPT, format_context
from vectorstore.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured result of a RAG query, ready to serialize into an API response."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence_score: float = 0.0
    response_time_ms: float = 0.0
    model_used: str = ""


class RAGChain:
    """Ties retrieval (ChromaStore) and generation (LLM provider) together."""

    def __init__(self, store: ChromaStore | None = None):
        self.store = store or ChromaStore()

    def _confidence_from_hits(self, hits: list[dict]) -> float:
        """Heuristic confidence: average similarity (1 - cosine distance) of retrieved
        chunks, clamped to [0, 1]. Not a calibrated probability — a relative signal
        for how well the retrieved context matches the question."""
        if not hits:
            return 0.0
        similarities = [max(0.0, 1.0 - hit["distance"]) for hit in hits]
        return round(sum(similarities) / len(similarities), 4)

    def query(self, question: str, top_k: int | None = None, provider: str | None = None) -> RAGResponse:
        """Run the full retrieve-then-generate pipeline for a single question."""
        start = time.perf_counter()
        top_k = top_k or settings.retrieval_top_k

        hits = self.store.similarity_search(question, top_k=top_k)

        if not hits:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RAGResponse(
                answer="I don't have enough information in the available drug labels to answer this question.",
                sources=[],
                confidence_score=0.0,
                response_time_ms=round(elapsed_ms, 2),
                model_used=provider or settings.llm_provider,
            )

        context = format_context(hits)
        llm = get_llm(provider)
        messages = QA_PROMPT.format_messages(context=context, question=question)

        logger.info("Querying LLM (provider=%s) with %d context chunks", provider or settings.llm_provider, len(hits))
        result = llm.invoke(messages)
        answer_text = result.content if hasattr(result, "content") else str(result)

        elapsed_ms = (time.perf_counter() - start) * 1000
        sources = [
            {
                "drug_name": hit["metadata"]["drug_name"],
                "section_name": hit["metadata"]["section_name"],
                "source_url": hit["metadata"]["source_url"],
            }
            for hit in hits
        ]

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            confidence_score=self._confidence_from_hits(hits),
            response_time_ms=round(elapsed_ms, 2),
            model_used=provider or settings.llm_provider,
        )


if __name__ == "__main__":
    chain = RAGChain()
    response = chain.query("What are the common side effects of ibuprofen?")
    logger.info("Answer: %s", response.answer)
    logger.info("Sources: %s", response.sources)
    logger.info("Confidence: %s, Time: %sms", response.confidence_score, response.response_time_ms)

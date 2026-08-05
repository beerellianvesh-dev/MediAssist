"""RAGAS-based evaluation of the RAG pipeline against a fixed test set.

Generates 20 Q&A pairs from ingested drug labels (question about a section,
expected answer = that section's text), runs them through the live RAG chain,
and scores answer relevancy, faithfulness, and context precision with RAGAS.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from rag.chain import RAGChain
from vectorstore.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

QUESTION_TEMPLATES = {
    "Indications and Usage": "What is {drug} used for?",
    "Warnings": "What are the warnings for {drug}?",
    "Adverse Reactions": "What are the adverse reactions of {drug}?",
    "Dosage and Administration": "What is the recommended dosage for {drug}?",
    "Drug Interactions": "What drug interactions should I know about for {drug}?",
    "Contraindications": "Who should not take {drug}?",
}


def build_test_set(store: ChromaStore, n: int = 20) -> list[dict]:
    """Sample n (question, ground_truth, drug) triples from ingested chunks."""
    records = store._collection.get(include=["documents", "metadatas"])
    seen_sections = set()
    test_set = []

    for doc, meta in zip(records["documents"], records["metadatas"]):
        key = (meta["drug_name"], meta["section_name"])
        if key in seen_sections:
            continue
        template = QUESTION_TEMPLATES.get(meta["section_name"])
        if not template:
            continue

        seen_sections.add(key)
        test_set.append(
            {
                "question": template.format(drug=meta["drug_name"]),
                "ground_truth": doc,
                "drug_name": meta["drug_name"],
                "section_name": meta["section_name"],
            }
        )
        if len(test_set) >= n:
            break

    return test_set


def run_evaluation(n_questions: int = 20, provider: str = "ollama") -> dict:
    """Run the test set through the RAG chain and score it with RAGAS."""
    store = ChromaStore()
    chain = RAGChain(store=store)

    test_set = build_test_set(store, n=n_questions)
    if not test_set:
        raise RuntimeError("No data in the vector store. Run scripts/ingest.py first.")

    logger.info("Running %d test questions through the RAG chain (provider=%s)...", len(test_set), provider)

    eval_rows = []
    for item in test_set:
        response = chain.query(item["question"], provider=provider)
        eval_rows.append(
            {
                "question": item["question"],
                "answer": response.answer,
                "contexts": [s["drug_name"] + " - " + s["section_name"] for s in response.sources],
                "ground_truth": item["ground_truth"],
                "response_time_ms": response.response_time_ms,
            }
        )

    scores = _score_with_ragas(eval_rows)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "n_questions": len(test_set),
        "avg_response_time_ms": round(sum(r["response_time_ms"] for r in eval_rows) / len(eval_rows), 2),
        "scores": scores,
        "rows": eval_rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{provider}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved evaluation results to %s", out_path)

    return result


def _score_with_ragas(eval_rows: list[dict], judge_provider: str = "ollama") -> dict:
    """Score with RAGAS, using our own LLM provider (Ollama by default) as the
    judge and the same local sentence-transformers model for its embeddings —
    so scoring stays fully free/local instead of silently requiring an OpenAI
    key. Falls back to a simple word-overlap heuristic if RAGAS itself fails
    (e.g. the judge model can't produce parseable structured output)."""
    try:
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas import RunConfig, evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        from rag.llm_provider import get_llm

        # RAGAS's default per-job timeout (180s) is tuned for hosted APIs; a
        # CPU-only local judge model needs much more headroom, and fewer
        # concurrent workers to avoid saturating one CPU with parallel calls.
        run_config = RunConfig(timeout=900, max_workers=2)

        judge_llm = get_llm(judge_provider)
        judge_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        dataset = Dataset.from_list(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"] or [""],
                    "ground_truth": r["ground_truth"],
                }
                for r in eval_rows
            ]
        )
        result = evaluate(
            dataset,
            metrics=[answer_relevancy, faithfulness, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )
        # EvaluationResult isn't a plain mapping (its __getitem__ indexes rows,
        # not metric names), so aggregate the per-row scores ourselves.
        import numpy as np

        metric_names = result.scores[0].keys()
        return {
            name: float(np.nanmean([row[name] for row in result.scores]))
            for name in metric_names
        }
    except Exception as exc:
        logger.warning("RAGAS scoring unavailable (%s); reporting heuristic overlap score instead.", exc)
        overlaps = []
        for r in eval_rows:
            gt_words = set(r["ground_truth"].lower().split())
            ans_words = set(r["answer"].lower().split())
            overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
            overlaps.append(overlap)
        return {"heuristic_word_overlap": round(sum(overlaps) / len(overlaps), 4)}


if __name__ == "__main__":
    run_evaluation()

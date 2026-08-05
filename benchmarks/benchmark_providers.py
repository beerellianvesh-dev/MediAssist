"""Compare available LLM providers on accuracy (RAGAS), latency, and estimated cost.

Only providers with credentials configured (or Ollama, which needs none) are
benchmarked. Results are saved as a JSON table and a bar chart PNG.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from benchmarks.evaluate import run_evaluation
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Rough published per-1K-token prices (USD) as of model release, blended
# input/output estimate. Ollama runs locally so its cost is $0.
ESTIMATED_COST_PER_1K_TOKENS = {
    "ollama": 0.0,
    "groq": 0.00059,   # llama3-70b-8192 on Groq free/pro tier, approx blended rate
    "openai": 0.00075,  # gpt-4o-mini blended estimate
    "anthropic": 0.0105,  # claude sonnet blended estimate
}
ASSUMED_TOKENS_PER_QUERY = 900  # context + question + answer, rough average


def _provider_available(provider: str) -> bool:
    if provider == "ollama":
        from rag.llm_provider import check_ollama_available

        return check_ollama_available()
    key_map = {
        "groq": settings.groq_api_key,
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
    }
    return bool(key_map.get(provider))


def run_benchmark(n_questions: int = 10) -> dict:
    providers = ["ollama", "groq", "openai", "anthropic"]
    available = [p for p in providers if _provider_available(p)]
    logger.info("Benchmarking available providers: %s", available)

    if not available:
        raise RuntimeError("No LLM providers are available. At minimum, start Ollama.")

    rows = []
    for provider in available:
        logger.info("Evaluating provider: %s", provider)
        try:
            result = run_evaluation(n_questions=n_questions, provider=provider)
        except Exception as exc:
            logger.error("Provider %s failed: %s", provider, exc)
            continue

        est_cost_per_query = (ASSUMED_TOKENS_PER_QUERY / 1000) * ESTIMATED_COST_PER_1K_TOKENS[provider]
        rows.append(
            {
                "provider": provider,
                "avg_response_time_ms": result["avg_response_time_ms"],
                "scores": result["scores"],
                "estimated_cost_per_query_usd": round(est_cost_per_query, 6),
            }
        )

    comparison = {"timestamp": datetime.now(timezone.utc).isoformat(), "providers": rows}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"provider_comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved comparison table to %s", json_path)

    _plot_comparison(rows, RESULTS_DIR / "provider_comparison.png")
    return comparison


def _plot_comparison(rows: list[dict], out_path: Path):
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    providers = [r["provider"] for r in rows]
    latencies = [r["avg_response_time_ms"] for r in rows]
    costs = [r["estimated_cost_per_query_usd"] for r in rows]

    axes[0].bar(providers, latencies, color="#0B5ED7")
    axes[0].set_title("Average Latency by Provider")
    axes[0].set_ylabel("ms")

    axes[1].bar(providers, costs, color="#1FAA59")
    axes[1].set_title("Estimated Cost per Query")
    axes[1].set_ylabel("USD")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved comparison chart to %s", out_path)


if __name__ == "__main__":
    run_benchmark()

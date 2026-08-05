"""Download drug label data from the openFDA API and store raw JSON to disk."""

import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Fields we care about for the RAG pipeline. openFDA returns others too,
# but these are the ones with meaningful clinical text.
RELEVANT_FIELDS = [
    "brand_name",
    "generic_name",
    "indications_and_usage",
    "warnings",
    "adverse_reactions",
    "dosage_and_administration",
    "drug_interactions",
    "contraindications",
]


def _extract_drug_name(result: dict) -> str:
    """Pull a human-readable drug name out of an openFDA result's openfda block."""
    openfda = result.get("openfda", {})
    brand = openfda.get("brand_name", [])
    generic = openfda.get("generic_name", [])
    if brand:
        return brand[0]
    if generic:
        return generic[0]
    return "unknown_drug"


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")[:80]


def fetch_drug_labels(limit: int = 100, batch_size: int = 25) -> list[dict]:
    """Fetch `limit` drug labels from openFDA, paginating in batches.

    Only labels that contain at least one of the RELEVANT_FIELDS are kept,
    since many openFDA entries are missing most clinical sections.
    """
    results: list[dict] = []
    skip = 0

    while len(results) < limit:
        params = {
            "limit": min(batch_size, limit - len(results)),
            "skip": skip,
        }
        if settings.openfda_api_key:
            params["api_key"] = settings.openfda_api_key

        logger.info("Fetching openFDA batch: skip=%s limit=%s", skip, params["limit"])
        response = requests.get(OPENFDA_URL, params=params, timeout=30)

        if response.status_code == 404:
            logger.warning("openFDA returned no more results at skip=%s", skip)
            break
        response.raise_for_status()

        payload = response.json()
        batch = payload.get("results", [])
        if not batch:
            break

        for record in batch:
            if any(field in record for field in RELEVANT_FIELDS):
                results.append(record)

        skip += batch_size
        time.sleep(0.3)  # be polite to the public API

    return results[:limit]


def save_raw_labels(labels: list[dict], out_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Save each label as its own JSON file, skipping ones already on disk.

    Filenames are derived from the drug name plus the openFDA set_id (when
    present) so re-running the download does not duplicate files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for record in labels:
        name = _extract_drug_name(record)
        set_id = record.get("id") or record.get("set_id") or ""
        filename = f"{_slugify(name)}_{set_id[:8] or 'na'}.json"
        path = out_dir / filename

        if path.exists():
            logger.info("Skipping existing file: %s", filename)
            continue

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        saved_paths.append(path)
        logger.info("Saved %s", filename)

    return saved_paths


def download_fda_labels(limit: int | None = None) -> list[Path]:
    """Entry point: fetch and persist openFDA drug labels."""
    limit = limit or settings.openfda_limit
    labels = fetch_drug_labels(limit=limit)
    logger.info("Fetched %d labels with relevant fields", len(labels))
    saved = save_raw_labels(labels)
    logger.info("Saved %d new label files (%d already existed)", len(saved), len(labels) - len(saved))
    return saved


def fetch_labels_for_drug(name: str, limit: int = 3) -> list[dict]:
    """Fetch labels matching a specific drug name (brand or generic) via openFDA search.

    Records whose openfda block is missing brand/generic name are tagged with
    `name` so `_extract_drug_name` doesn't fall back to "unknown_drug" for a
    drug we explicitly searched for.
    """
    search = f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"'
    params = {"search": search, "limit": limit}
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key

    logger.info("Searching openFDA for drug=%s", name)
    response = requests.get(OPENFDA_URL, params=params, timeout=30)
    if response.status_code == 404:
        logger.warning("openFDA returned no results for %s", name)
        return []
    response.raise_for_status()

    batch = [r for r in response.json().get("results", []) if any(f in r for f in RELEVANT_FIELDS)]
    for record in batch:
        if _extract_drug_name(record) == "unknown_drug":
            record.setdefault("openfda", {}).setdefault("generic_name", [name.title()])

    time.sleep(0.3)  # be polite to the public API
    return batch


def download_labels_for_drugs(names: list[str], per_drug_limit: int = 3) -> list[Path]:
    """Fetch and persist openFDA labels for a specific list of well-known drug names.

    Used to seed the vector store with recognizable, demo-friendly drugs
    instead of relying on openFDA's default (largely OTC/random) ordering.
    """
    all_labels: list[dict] = []
    for name in names:
        all_labels.extend(fetch_labels_for_drug(name, limit=per_drug_limit))

    logger.info("Fetched %d labels for %d requested drugs", len(all_labels), len(names))
    saved = save_raw_labels(all_labels)
    logger.info("Saved %d new label files (%d already existed)", len(saved), len(all_labels) - len(saved))
    return saved


if __name__ == "__main__":
    download_fda_labels()

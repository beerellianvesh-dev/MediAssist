"""Run the full ingestion pipeline: download -> chunk -> embed & store.

Usage:
    python scripts/ingest.py [--limit N] [--skip-download] [--pdf-dir PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import chunk_raw_directory
from ingestion.fda_downloader import download_fda_labels, download_labels_for_drugs
from ingestion.pdf_extractor import extract_pdfs_from_folder
from vectorstore.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest openFDA drug labels into ChromaDB.")
    parser.add_argument("--limit", type=int, default=None, help="Number of FDA labels to download.")
    parser.add_argument("--skip-download", action="store_true", help="Skip openFDA download, reuse data/raw.")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Optional folder of PDFs to also ingest.")
    parser.add_argument(
        "--drugs", type=str, default=None,
        help="Comma-separated list of specific drug names to fetch instead of the default random sample.",
    )
    parser.add_argument(
        "--per-drug-limit", type=int, default=3,
        help="Max labels to fetch per drug name when using --drugs.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        if args.drugs:
            names = [n.strip() for n in args.drugs.split(",") if n.strip()]
            logger.info("Step 1/3: Downloading targeted drug labels for %s...", names)
            download_labels_for_drugs(names, per_drug_limit=args.per_drug_limit)
        else:
            logger.info("Step 1/3: Downloading drug labels from openFDA...")
            download_fda_labels(limit=args.limit)
    else:
        logger.info("Step 1/3: Skipped (--skip-download)")

    if args.pdf_dir:
        logger.info("Extracting text from PDFs in %s", args.pdf_dir)
        extract_pdfs_from_folder(Path(args.pdf_dir))

    logger.info("Step 2/3: Chunking documents...")
    chunks = chunk_raw_directory()
    logger.info("Produced %d chunks", len(chunks))

    logger.info("Step 3/3: Embedding and storing in ChromaDB...")
    store = ChromaStore()
    added = store.add_documents([c.to_dict() for c in chunks])
    logger.info("Added %d new chunks to the vector store", added)
    logger.info("Collection stats: %s", store.get_collection_stats())


if __name__ == "__main__":
    main()

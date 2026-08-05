"""Semantic chunking of drug label documents.

Chunking is section-aware: each clinical section (indications, warnings,
etc.) is treated as its own semantic unit first, then sub-split by token
count only if it exceeds the target chunk size. This keeps related content
together instead of blindly cutting at a fixed character offset.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from ingestion.fda_downloader import RELEVANT_FIELDS, _extract_drug_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Human-readable section names for citation purposes.
SECTION_LABELS = {
    "indications_and_usage": "Indications and Usage",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse Reactions",
    "dosage_and_administration": "Dosage and Administration",
    "drug_interactions": "Drug Interactions",
    "contraindications": "Contraindications",
}


@dataclass
class Chunk:
    """A single chunk of text ready for embedding, with citation metadata."""

    text: str
    drug_name: str
    section_name: str
    source_url: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "drug_name": self.drug_name,
            "section_name": self.section_name,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def _flatten_field(value) -> str:
    """openFDA fields are usually list[str]; join them into one string."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter, good enough for clinical prose."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def _chunk_text_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into chunks of ~chunk_size tokens with `overlap` token overlap,
    breaking on sentence boundaries where possible."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(_ENCODING.encode(sentence))

        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Build overlap: keep trailing sentences worth ~`overlap` tokens.
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = len(_ENCODING.encode(s))
                if overlap_tokens + s_tokens > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def chunk_fda_label(
    record: dict,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Turn one raw openFDA label record into a list of citation-tagged Chunks."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    drug_name = _extract_drug_name(record)
    set_id = record.get("id") or record.get("set_id") or ""
    source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else ""

    chunks: list[Chunk] = []
    for field_name in RELEVANT_FIELDS:
        if field_name not in record:
            continue

        section_text = _flatten_field(record[field_name]).strip()
        if not section_text:
            continue

        section_label = SECTION_LABELS.get(field_name, field_name)
        text_pieces = _chunk_text_by_tokens(section_text, chunk_size, overlap)

        for i, piece in enumerate(text_pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    drug_name=drug_name,
                    section_name=section_label,
                    source_url=source_url,
                    chunk_index=i,
                    metadata={"field": field_name, "set_id": set_id},
                )
            )

    return chunks


def chunk_raw_directory(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> list[Chunk]:
    """Chunk every JSON label file in `raw_dir` and write the result to `processed_dir`."""
    raw_dir = raw_dir or Path(__file__).resolve().parent.parent / "data" / "raw"
    processed_dir = processed_dir or Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    json_files = sorted(raw_dir.glob("*.json"))
    logger.info("Found %d raw label files", len(json_files))

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        chunks = chunk_fda_label(record)
        all_chunks.extend(chunks)

    out_path = processed_dir / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")

    logger.info("Wrote %d chunks to %s", len(all_chunks), out_path)
    return all_chunks


if __name__ == "__main__":
    chunk_raw_directory()

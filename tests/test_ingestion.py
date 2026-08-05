"""Tests for the ingestion pipeline: FDA parsing and chunking."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import chunk_fda_label, _chunk_text_by_tokens, _split_sentences
from ingestion.fda_downloader import _extract_drug_name, _slugify

SAMPLE_RECORD = {
    "id": "abc12345-full",
    "set_id": "abc12345",
    "openfda": {"brand_name": ["Advil"], "generic_name": ["ibuprofen"]},
    "indications_and_usage": [
        "Advil is indicated for the temporary relief of minor aches and pains. " * 20
    ],
    "warnings": ["Stomach bleeding warning: NSAIDs may cause severe stomach bleeding."],
}


def test_extract_drug_name_prefers_brand():
    assert _extract_drug_name(SAMPLE_RECORD) == "Advil"


def test_extract_drug_name_falls_back_to_generic():
    record = {"openfda": {"generic_name": ["ibuprofen"]}}
    assert _extract_drug_name(record) == "ibuprofen"


def test_extract_drug_name_unknown_when_missing():
    assert _extract_drug_name({}) == "unknown_drug"


def test_slugify_basic():
    assert _slugify("Advil PM (Extra Strength)!") == "advil_pm__extra_strength"


def test_split_sentences():
    sentences = _split_sentences("Take one tablet daily. Do not exceed 4 tablets in 24 hours!")
    assert len(sentences) == 2


def test_chunk_text_respects_token_budget():
    long_text = "This is a clinical sentence about dosage and administration. " * 100
    chunks = _chunk_text_by_tokens(long_text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_fda_label_produces_metadata():
    chunks = chunk_fda_label(SAMPLE_RECORD, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.drug_name == "Advil"
        assert chunk.section_name in {"Indications and Usage", "Warnings"}
        assert "dailymed.nlm.nih.gov" in chunk.source_url
        assert chunk.text.strip() != ""


def test_chunk_fda_label_skips_missing_fields():
    record = {"openfda": {"brand_name": ["Tylenol"]}, "warnings": ["Liver damage warning."]}
    chunks = chunk_fda_label(record)
    sections = {c.section_name for c in chunks}
    assert sections == {"Warnings"}

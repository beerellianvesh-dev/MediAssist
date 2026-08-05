"""Extract text from user-supplied PDF drug labels."""

import logging
from pathlib import Path

from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a single PDF file, page by page."""
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)
        else:
            logger.warning("No extractable text on page %d of %s", page_num, pdf_path.name)
    return "\n\n".join(pages_text)


def extract_pdfs_from_folder(folder: Path) -> dict[str, str]:
    """Extract text from every PDF in `folder`, returning {filename: text}."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder does not exist: {folder}")

    results: dict[str, str] = {}
    for pdf_path in sorted(folder.glob("*.pdf")):
        logger.info("Extracting text from %s", pdf_path.name)
        try:
            results[pdf_path.stem] = extract_text_from_pdf(pdf_path)
        except Exception as exc:  # malformed/encrypted PDFs shouldn't kill the batch
            logger.error("Failed to extract %s: %s", pdf_path.name, exc)
    return results


if __name__ == "__main__":
    import sys

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/pdfs")
    extracted = extract_pdfs_from_folder(folder)
    logger.info("Extracted text from %d PDF(s)", len(extracted))

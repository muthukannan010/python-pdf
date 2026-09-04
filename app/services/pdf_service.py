import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

from app.utils.config import settings

logger = logging.getLogger(__name__)

_MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


class PDFProcessingError(Exception):
    """Raised when something goes wrong processing a PDF."""
    pass


def sanitize_filename(name: str) -> str:
    """Make a filename safe for the filesystem."""
    name = unicodedata.normalize("NFKD", name)
    # strip out anything that's not alphanumeric, whitespace, hyphens, or dots
    name = re.sub(r"[^\w\s\-.]", "", name, flags=re.ASCII)
    name = re.sub(r"\s+", "_", name.strip())
    name = name.lstrip(".")
    return name or "document.pdf"


def validate_pdf_bytes(data: bytes, filename: str):
    """Check that the uploaded bytes are actually a valid PDF and not too large."""
    if len(data) == 0:
        raise PDFProcessingError("Uploaded file is empty.")

    if len(data) > _MAX_BYTES:
        raise PDFProcessingError(
            f"File size {len(data) / 1_048_576:.1f} MB exceeds the "
            f"{settings.MAX_FILE_SIZE_MB} MB limit."
        )

    if not data[:4] == b"%PDF":
        raise PDFProcessingError(
            f"'{filename}' does not appear to be a valid PDF file."
        )


def save_pdf(data: bytes, document_id: str) -> Path:
    """Save uploaded PDF bytes to disk."""
    settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = settings.PDF_DIR / f"{document_id}.pdf"
    dest.write_bytes(data)
    logger.info("Saved PDF to %s (%d bytes)", dest, len(data))
    return dest


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from each page of a PDF using PyMuPDF.
    Returns a list of dicts with keys: document, page, text, words
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFProcessingError(f"Cannot open PDF '{pdf_path.name}': {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise PDFProcessingError(
            f"'{pdf_path.name}' is password-protected and cannot be processed."
        )

    total_pages = len(doc)
    pages = []
    for i in range(total_pages):
        page = doc.load_page(i)
        
        # Get raw words: list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        raw_words = page.get_text("words")
        
        # Sort words top-to-bottom, left-to-right loosely (blocks then lines then words)
        raw_words.sort(key=lambda w: (w[5], w[6], w[7]))
        
        text = " ".join(w[4] for w in raw_words).strip()
        words = [{"text": w[4], "bbox": [w[0], w[1], w[2], w[3]]} for w in raw_words]
        
        if text:
            pages.append({
                "document": pdf_path.name,
                "page": i + 1,   # 1-based
                "text": text,
                "words": words,
            })

    doc.close()

    if not pages:
        raise PDFProcessingError(
            f"'{pdf_path.name}' contains no extractable text. "
            "It may be a scanned image without OCR."
        )

    logger.info(
        "Extracted text from %d/%d pages of '%s'",
        len(pages), total_pages, pdf_path.name,
    )
    return pages


def process_pdf_upload(data: bytes, original_filename: str) -> tuple[str, Path, List[Dict[str, Any]]]:
    """
    Full upload pipeline: validate -> save -> extract.
    Returns (document_id, pdf_path, pages)
    """
    safe_name = sanitize_filename(original_filename)
    document_id = Path(safe_name).stem

    logger.info("Processing upload: original='%s', id='%s'", original_filename, document_id)

    validate_pdf_bytes(data, original_filename)
    pdf_path = save_pdf(data, document_id)
    pages = extract_text_from_pdf(pdf_path)

    return document_id, pdf_path, pages

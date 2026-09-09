import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import io

from app.utils.config import settings

logger = logging.getLogger(__name__)

_MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


class PDFProcessingError(Exception):
    """Raised when something goes wrong processing a PDF."""
    pass


def sanitize_filename(name: str) -> str:
    """Make a filename safe for the filesystem."""
    name = unicodedata.normalize("NFKD", name)

    # Strip out anything that's not alphanumeric, whitespace,
    # hyphens, underscores, or dots.
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

    logger.info(
        "Saved PDF to %s (%d bytes)",
        dest,
        len(data),
    )

    return dest

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from each page of a PDF.

    Normal PDF:
        Uses PyMuPDF text extraction.

    Scanned PDF:
        Automatically falls back to Tesseract OCR.

    Returns a list of dictionaries containing:
        document
        page
        text
        words
        extraction_method
    """

    try:
        doc = fitz.open(str(pdf_path))

    except Exception as exc:
        raise PDFProcessingError(
            f"Cannot open PDF '{pdf_path.name}': {exc}"
        ) from exc

    if doc.needs_pass:
        doc.close()

        raise PDFProcessingError(
            f"'{pdf_path.name}' is password-protected and cannot be processed."
        )

    total_pages = len(doc)

    pages = []

    for i in range(total_pages):

        page = doc.load_page(i)

        # ---------------------------------------------------------
        # STEP 1: Try normal PDF text extraction
        # ---------------------------------------------------------

        raw_words = page.get_text("words")

        # Sort words:
        # block → line → word
        raw_words.sort(
            key=lambda w: (
                w[5],
                w[6],
                w[7],
            )
        )

        text = " ".join(
            w[4]
            for w in raw_words
        ).strip()

        words = [
            {
                "text": w[4],
                "bbox": [
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                ],
            }
            for w in raw_words
        ]

        extraction_method = "native"

        # ---------------------------------------------------------
        # STEP 2: If no text exists, use Tesseract OCR
        # ---------------------------------------------------------

        if not text:

            logger.info(
                "No native text found on page %d of '%s'. "
                "Using Tesseract fallback.",
                i + 1,
                pdf_path.name,
            )

            text, words = extract_text_with_ocr(page)

            extraction_method = "ocr"

        # ---------------------------------------------------------
        # STEP 3: Store page if text was successfully extracted
        # ---------------------------------------------------------

        if text:

            pages.append(
                {
                    "document": pdf_path.name,
                    "page": i + 1,
                    "text": text,
                    "words": words,
                    "extraction_method": extraction_method,
                }
            )

            logger.info(
                "Page %d/%d extracted using %s",
                i + 1,
                total_pages,
                extraction_method,
            )

        else:

            logger.warning(
                "No text could be extracted from page %d/%d of '%s'",
                i + 1,
                total_pages,
                pdf_path.name,
            )

    doc.close()

    # -------------------------------------------------------------
    # STEP 4: Fail only if BOTH native extraction and OCR failed
    # -------------------------------------------------------------

    if not pages:

        raise PDFProcessingError(
            f"'{pdf_path.name}' contains no extractable text even after "
            "Tesseract OCR processing."
        )

    logger.info(
        "Extracted text from %d/%d pages of '%s'",
        len(pages),
        total_pages,
        pdf_path.name,
    )

    return pages


def process_pdf_upload(
    data: bytes,
    original_filename: str,
) -> tuple[str, Path, List[Dict[str, Any]]]:
    """
    Full upload pipeline:

        validate
            ↓
        save
            ↓
        extract native text / Tesseract OCR

    Returns:
        (document_id, pdf_path, pages)
    """

    safe_name = sanitize_filename(original_filename)

    document_id = Path(safe_name).stem

    logger.info(
        "Processing upload: original='%s', id='%s'",
        original_filename,
        document_id,
    )

    validate_pdf_bytes(
        data,
        original_filename,
    )

    pdf_path = save_pdf(
        data,
        document_id,
    )

    pages = extract_text_from_pdf(
        pdf_path
    )

    return (
        document_id,
        pdf_path,
        pages,
    )

def extract_text_with_ocr(page: fitz.Page) -> tuple[str, List[Dict[str, Any]]]:
    """Fallback OCR using Tesseract for scanned PDFs."""
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    words = []
    text_parts = []
    
    for i in range(len(data['text'])):
        word_text = data['text'][i].strip()
        conf = float(data['conf'][i])
        
        if conf >= 0 and word_text:
            x0 = data['left'][i] / 2.0
            y0 = data['top'][i] / 2.0
            w = data['width'][i] / 2.0
            h = data['height'][i] / 2.0
            x1 = x0 + w
            y1 = y0 + h
            
            words.append({
                "text": word_text,
                "bbox": [x0, y0, x1, y1],
                "confidence": conf
            })
            text_parts.append(word_text)
            
    return " ".join(text_parts), words
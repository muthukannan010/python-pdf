import logging
from typing import Any, Dict, List

from app.utils.config import settings

logger = logging.getLogger(__name__)


def _make_chunk_id(document_id: str, page: int, index: int) -> str:
    """Build a readable chunk identifier like doc1_page0003_chunk001."""
    return f"{document_id}_page{page:04d}_chunk{index:03d}"


def chunk_page_text(
    text: str,
    document_id: str,
    document_name: str,
    page_number: int,
    source_path: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    chunk_index_start: int = 0,
) -> List[Dict[str, Any]]:
    """Split a page's text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    idx = chunk_index_start

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append({
                "chunk_id": _make_chunk_id(document_id, page_number, idx),
                "document_id": document_id,
                "document_name": document_name,
                "page_number": page_number,
                "text": chunk_text,
                "source_path": source_path,
            })
            idx += 1

        if end == len(words):
            break
        start += step

    return chunks


def create_chunks_from_pages(
    pages: List[Dict[str, Any]],
    document_id: str,
    document_name: str,
    source_path: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Take the extracted pages and break them all into chunks."""
    all_chunks = []
    global_idx = 0

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        if not text.strip():
            continue

        page_chunks = chunk_page_text(
            text=text,
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            source_path=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_index_start=global_idx,
        )
        all_chunks.extend(page_chunks)
        global_idx += len(page_chunks)

    logger.info(
        "Created %d chunks from %d pages for document '%s'",
        len(all_chunks), len(pages), document_name,
    )
    return all_chunks

"""API routes for document upload, search, and management."""

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.database.sqlite_db import SQLiteDB
from app.database.vector_store import VectorStore
from app.models.schemas import (
    DeleteResponse, DocumentInfo, DocumentListResponse, HealthResponse,
    SearchRequest, SearchResponse, SearchResult, UploadResponse,
)
from app.services.chunk_service import create_chunks_from_pages
from app.services.embedding_service import encode_texts, is_model_loaded
from app.services.hybrid_search import HybridSearchService
from app.services.pdf_service import PDFProcessingError, process_pdf_upload
from app.utils.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# pull shared instances from app.state
def _get_db(request: Request) -> SQLiteDB:
    return request.app.state.db

def _get_vs(request: Request) -> VectorStore:
    return request.app.state.vector_store

def _get_hybrid(request: Request) -> HybridSearchService:
    return request.app.state.hybrid_service


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(
    db: SQLiteDB = Depends(_get_db),
    vs: VectorStore = Depends(_get_vs),
):
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        faiss_index_size=vs.size,
        sqlite_document_count=db.get_document_count(),
    )


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF",
)
async def upload_document(
    file: UploadFile = File(...),
    db: SQLiteDB = Depends(_get_db),
    vs: VectorStore = Depends(_get_vs),
):
    # quick mime type check
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are accepted.",
            )

    data = await file.read()
    original_name = file.filename or "document.pdf"

    try:
        document_id, pdf_path, pages = process_pdf_upload(data, original_name)
    except PDFProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # check for duplicates
    if db.document_exists(document_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A document with id '{document_id}' already exists. Delete it first.",
        )

    source_path = f"data/pdfs/{document_id}.pdf"

    # chunk the text
    chunks = create_chunks_from_pages(
        pages=pages,
        document_id=document_id,
        document_name=original_name,
        source_path=source_path,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be chunked from the document.",
        )

    # generate embeddings (runs locally)
    texts = [c["text"] for c in chunks]
    embeddings = encode_texts(texts)

    # store in FAISS + SQLite
    vs.add_vectors(embeddings=embeddings, chunk_metas=chunks)
    db.insert_chunks(chunks)
    db.insert_document(
        document_id=document_id,
        document_name=original_name,
        page_count=len(pages),
        chunk_count=len(chunks),
        source_path=source_path,
    )

    logger.info("Indexed '%s': %d pages, %d chunks.", original_name, len(pages), len(chunks))

    return UploadResponse(
        document_id=document_id,
        document_name=original_name,
        page_count=len(pages),
        chunk_count=len(chunks),
        message=f"Successfully uploaded and indexed '{original_name}'.",
    )


@router.get("/documents", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(db: SQLiteDB = Depends(_get_db)):
    docs = db.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document",
)
async def delete_document(
    document_id: str,
    db: SQLiteDB = Depends(_get_db),
    vs: VectorStore = Depends(_get_vs),
):
    if not db.document_exists(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Document '{document_id}' not found.")

    # remove PDF file
    pdf_path = settings.PDF_DIR / f"{document_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    # remove from indexes
    vs.delete_by_document(document_id)
    db.delete_document(document_id)

    logger.info("Deleted document '%s'.", document_id)
    return DeleteResponse(document_id=document_id,
                          message=f"Document '{document_id}' deleted successfully.")


@router.get(
    "/documents/{document_id}/view",
    summary="Serve PDF for viewing",
    response_class=FileResponse,
)
async def view_document(document_id: str, db: SQLiteDB = Depends(_get_db)):
    """Return the PDF file so the browser can render it (frontend appends #page=N)."""
    if not db.document_exists(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Document '{document_id}' not found.")

    pdf_path = settings.PDF_DIR / f"{document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="PDF file not found on disk.")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{document_id}.pdf",
        headers={"Content-Disposition": f'inline; filename="{document_id}.pdf"'},
    )


@router.post("/search", response_model=SearchResponse, summary="Hybrid search")
async def search(
    request_body: SearchRequest,
    hybrid: HybridSearchService = Depends(_get_hybrid),
):
    if not request_body.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Search query must not be empty.")

    kw_weight = request_body.keyword_weight
    sem_weight = (1.0 - kw_weight) if kw_weight is not None else None

    result = hybrid.search(
        query=request_body.query,
        top_k=request_body.top_k,
        keyword_weight=kw_weight,
        semantic_weight=sem_weight,
    )

    search_results = [
        SearchResult(
            chunk_id=r["chunk_id"],
            document_name=r["document_name"],
            page_number=r["page_number"],
            score=round(r["score"], 4),
            keyword_score=round(r["keyword_score"], 4),
            semantic_score=round(r["semantic_score"], 4),
            text=r["text"],
            source_path=r["source_path"],
            matched_keywords=r.get("matched_keywords", []),
            unmatched_keywords=r.get("unmatched_keywords", []),
            locations=r.get("locations", []),
        )
        for r in result["results"]
    ]

    return SearchResponse(
        query=result["query"],
        results=search_results,
        total_results=result["total_results"],
        search_time_ms=result["search_time_ms"],
    )

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Document related schemas ---

class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    page_count: int
    chunk_count: int
    source_path: str
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    page_count: int
    chunk_count: int
    message: str


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# --- Search related schemas ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    keyword_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Keyword weight (0-1). Semantic weight = 1 - this value.",
    )


class HighlightLocation(BaseModel):
    text: str
    bbox: List[float]


class SearchResult(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    score: float
    keyword_score: float
    semantic_score: float
    text: str
    source_path: str
    matched_keywords: List[str] = Field(default_factory=list)
    unmatched_keywords: List[str] = Field(default_factory=list)
    locations: List[HighlightLocation] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    faiss_index_size: int
    sqlite_document_count: int
    version: str = "1.0.0"

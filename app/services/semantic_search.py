"""Semantic search using FAISS vector similarity."""

import logging
from typing import Any, Dict, List

from app.database.vector_store import VectorStore
from app.services.embedding_service import encode_query
from app.utils.config import settings

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """Meaning-based search over the FAISS vector index."""

    def __init__(self, vector_store: VectorStore):
        self._vs = vector_store

    def search(self, query: str, top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        """Embed the query and find the most similar chunks."""
        if not query.strip():
            return []

        query_vec = encode_query(query)
        results = self._vs.search(query_vec, top_k=top_k)

        # scale scores to 0-100 range to match keyword scores
        for r in results:
            if "semantic_score" in r:
                r["semantic_score"] *= 100.0

        logger.debug("SemanticSearch: query=%r → %d results", query, len(results))
        return results

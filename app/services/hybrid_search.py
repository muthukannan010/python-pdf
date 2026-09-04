"""
Hybrid search - merges keyword and semantic results into a single ranked list.

The final score for each chunk is:
    score = keyword_weight * keyword_score + semantic_weight * semantic_score
"""

import logging
import time
from typing import Any, Dict, List, Optional

from app.services.keyword_search import KeywordSearchService
from app.services.semantic_search import SemanticSearchService
from app.utils.config import settings

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Combines keyword and semantic search with configurable weights."""

    def __init__(self, keyword_service: KeywordSearchService, semantic_service: SemanticSearchService):
        self._kw = keyword_service
        self._sem = semantic_service

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run both searches, merge by chunk_id, compute hybrid score, return ranked results."""
        t0 = time.perf_counter()

        # figure out weights (use defaults if not provided)
        kw_weight = keyword_weight if keyword_weight is not None else settings.KEYWORD_WEIGHT
        sem_weight = semantic_weight if semantic_weight is not None else settings.SEMANTIC_WEIGHT

        # clamp and normalize so they add up to 1.0
        kw_weight = max(0.0, min(1.0, kw_weight))
        sem_weight = max(0.0, min(1.0, sem_weight))
        total = kw_weight + sem_weight
        if total > 0:
            kw_weight /= total
            sem_weight /= total

        # fetch extra candidates to get better coverage after merging
        fetch_k = max(top_k * 3, 20)

        kw_results = self._kw.search(query, top_k=fetch_k)
        sem_results = self._sem.search(query, top_k=fetch_k)

        def _truncate(text: str) -> str:
            """Shorten text for semantic-only results that have no highlighting."""
            words = text.split()
            if len(words) <= settings.SNIPPET_LENGTH:
                return text
            return " ".join(words[: settings.SNIPPET_LENGTH]) + "..."

        # merge results by chunk_id
        merged: Dict[str, Dict[str, Any]] = {}

        for r in kw_results:
            cid = r["chunk_id"]
            merged[cid] = {
                "chunk_id": cid,
                "document_name": r["document_name"],
                "page_number": r["page_number"],
                "text": r["text"],
                "source_path": r["source_path"],
                "keyword_score": r.get("keyword_score", 0.0),
                "semantic_score": 0.0,
                "matched_keywords": r.get("matched_keywords", []),
                "unmatched_keywords": r.get("unmatched_keywords", []),
                "locations": r.get("locations", []),
            }

        for r in sem_results:
            cid = r["chunk_id"]
            if cid in merged:
                # chunk already found by keyword search, just add semantic score
                merged[cid]["semantic_score"] = r.get("semantic_score", 0.0)
            else:
                # fetch locations from sqlite since FAISS doesn't store them
                chunk_meta = self._kw._db.get_chunk_by_id(cid)
                locs = chunk_meta.get("locations", []) if chunk_meta else []
                merged[cid] = {
                    "chunk_id": cid,
                    "document_name": r.get("document_name", ""),
                    "page_number": r.get("page_number", 0),
                    "text": _truncate(r.get("text", "")),
                    "source_path": r.get("source_path", ""),
                    "keyword_score": 0.0,
                    "semantic_score": r.get("semantic_score", 0.0),
                    "matched_keywords": [],
                    "unmatched_keywords": [],
                    "locations": locs,
                }

        # compute final hybrid score and tag the search type
        ranked = []
        for item in merged.values():
            item["score"] = round(
                kw_weight * item["keyword_score"] + sem_weight * item["semantic_score"],
                6,
            )

            # label how this result was found
            if item["keyword_score"] > 0 and item["semantic_score"] > 0:
                item["search_type"] = "Keyword + Semantic"
            elif item["keyword_score"] > 0:
                item["search_type"] = "Keyword"
            else:
                item["search_type"] = "Semantic"

            ranked.append(item)

        # sort by score, break ties with keyword score
        ranked.sort(key=lambda x: (x["score"], x["keyword_score"]), reverse=True)
        ranked = ranked[:top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "HybridSearch: query=%r → %d results (%.1f ms)",
            query, len(ranked), elapsed_ms,
        )

        return {
            "query": query,
            "results": ranked,
            "total_results": len(ranked),
            "search_time_ms": round(elapsed_ms, 2),
        }

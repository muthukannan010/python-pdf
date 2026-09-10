"""
Hybrid search - merges keyword and semantic results into a single ranked list.

Final score:
    score = keyword_weight * keyword_score
          + semantic_weight * semantic_score

Keyword score is expected to be 0-100.
Semantic score is expected to be 0-100.
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

    def __init__(
        self,
        keyword_service: KeywordSearchService,
        semantic_service: SemanticSearchService,
    ):
        self._kw = keyword_service
        self._sem = semantic_service

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run keyword + semantic search and return ranked results."""

        t0 = time.perf_counter()

        # ---------------------------------------------------------
        # 1. Get search weights
        # ---------------------------------------------------------

        kw_weight = (
            keyword_weight
            if keyword_weight is not None
            else settings.KEYWORD_WEIGHT
        )

        sem_weight = (
            semantic_weight
            if semantic_weight is not None
            else settings.SEMANTIC_WEIGHT
        )

        # Clamp weights to 0-1.
        kw_weight = max(0.0, min(1.0, kw_weight))
        sem_weight = max(0.0, min(1.0, sem_weight))

        # Normalize weights so they add up to 1.
        total = kw_weight + sem_weight

        if total > 0:
            kw_weight /= total
            sem_weight /= total

        # ---------------------------------------------------------
        # 2. Fetch extra candidates
        # ---------------------------------------------------------

        fetch_k = max(top_k * 3, 20)

        kw_results = self._kw.search(
            query,
            top_k=fetch_k,
        )

        sem_results = self._sem.search(
            query,
            top_k=fetch_k,
        )

        # ---------------------------------------------------------
        # 3. Helper for snippets
        # ---------------------------------------------------------

        def _truncate(text: str) -> str:
            words = text.split()

            if len(words) <= settings.SNIPPET_LENGTH:
                return text

            return (
                " ".join(words[: settings.SNIPPET_LENGTH])
                + "..."
            )

        # ---------------------------------------------------------
        # 4. Merge keyword + semantic results
        # ---------------------------------------------------------

        merged: Dict[str, Dict[str, Any]] = {}

        # Keyword results
        for r in kw_results:

            cid = r["chunk_id"]

            merged[cid] = {
                "chunk_id": cid,
                "document_id": r["document_id"],
                "document_name": r["document_name"],
                "page_number": r["page_number"],
                "text": r["text"],
                "source_path": r["source_path"],

                # Scores
                "keyword_score": float(
                    r.get("keyword_score", 0.0)
                ),
                "semantic_score": 0.0,

                # Keyword information
                "matched_keywords": r.get(
                    "matched_keywords",
                    [],
                ),
                "unmatched_keywords": r.get(
                    "unmatched_keywords",
                    [],
                ),

                # Source locations
                "locations": r.get(
                    "locations",
                    [],
                ),

                # OCR information
                "ocr_confidence": None,
            }

        # Semantic results
        for r in sem_results:

            cid = r["chunk_id"]

            if cid in merged:

                merged[cid]["semantic_score"] = float(
                    r.get("semantic_score", 0.0)
                )

            else:

                # FAISS does not contain all metadata,
                # so retrieve it from SQLite.
                chunk_meta = self._kw._db.get_chunk_by_id(cid)

                locs = (
                    chunk_meta.get("locations", [])
                    if chunk_meta
                    else []
                )

                # Get OCR confidence from locations.
                ocr_confidence = self._calculate_ocr_confidence(
                    locs
                )

                merged[cid] = {
                    "chunk_id": cid,
                    "document_id": r.get("document_id", ""),
                    "document_name": r.get(
                        "document_name",
                        "",
                    ),
                    "page_number": r.get(
                        "page_number",
                        0,
                    ),
                    "text": _truncate(
                        r.get("text", "")
                    ),
                    "source_path": r.get(
                        "source_path",
                        "",
                    ),

                    "keyword_score": 0.0,

                    "semantic_score": float(
                        r.get("semantic_score", 0.0)
                    ),

                    "matched_keywords": [],
                    "unmatched_keywords": [],

                    "locations": locs,

                    "ocr_confidence": ocr_confidence,
                }

        # ---------------------------------------------------------
        # 5. Calculate hybrid score
        # ---------------------------------------------------------

        ranked: List[Dict[str, Any]] = []

        for item in merged.values():

            keyword_score = float(
                item.get("keyword_score", 0.0)
            )

            semantic_score = float(
                item.get("semantic_score", 0.0)
            )

            final_score = (
                kw_weight * keyword_score
                + sem_weight * semantic_score
            )

            # Keep the final score in 0-100 range.
            final_score = max(
                0.0,
                min(100.0, final_score),
            )

            item["score"] = round(
                final_score,
                2,
            )

            # Explicit name for UI/API.
            item["relevance_percentage"] = round(
                final_score,
                2,
            )

            # -----------------------------------------------------
            # Search type
            # -----------------------------------------------------

            if keyword_score > 0 and semantic_score > 0:
                item["search_type"] = (
                    "Keyword + Semantic"
                )

            elif keyword_score > 0:
                item["search_type"] = "Keyword"

            else:
                item["search_type"] = "Semantic"

            # -----------------------------------------------------
            # OCR confidence
            # -----------------------------------------------------

            if item.get("ocr_confidence") is None:
                item["ocr_confidence"] = (
                    self._calculate_ocr_confidence(
                        item.get("locations", [])
                    )
                )

            ranked.append(item)

        # ---------------------------------------------------------
        # 6. Sort results
        # ---------------------------------------------------------

        ranked.sort(
            key=lambda x: (
                x["score"],
                x["keyword_score"],
                x["semantic_score"],
            ),
            reverse=True,
        )

        ranked = ranked[:top_k]

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000

        logger.info(
            "HybridSearch: query=%r → %d results (%.1f ms)",
            query,
            len(ranked),
            elapsed_ms,
        )

        # ---------------------------------------------------------
        # 7. Return response
        # ---------------------------------------------------------

        return {
            "query": query,

            "results": ranked,

            "total_results": len(ranked),

            "search_time_ms": round(
                elapsed_ms,
                2,
            ),

            "weights": {
                "keyword": round(
                    kw_weight,
                    3,
                ),
                "semantic": round(
                    sem_weight,
                    3,
                ),
            },
        }

    @staticmethod
    def _calculate_ocr_confidence(
        locations: List[Dict[str, Any]],
    ) -> Optional[float]:
        """
        Calculate average OCR confidence.

        Returns:
            0-100 percentage, or None when OCR
            confidence information is unavailable.
        """

        confidences = []

        for location in locations:

            confidence = location.get(
                "confidence"
            )

            if confidence is None:
                continue

            try:
                confidence = float(confidence)

                # Tesseract confidence is normally 0-100, but handle if it's 0-1.
                if confidence <= 1:
                    confidence *= 100

                confidence = max(
                    0.0,
                    min(100.0, confidence),
                )

                confidences.append(
                    confidence
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        if not confidences:
            return None

        return round(
            sum(confidences) / len(confidences),
            2,
        )
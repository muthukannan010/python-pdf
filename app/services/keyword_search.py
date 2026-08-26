"""Keyword search - computes what % of query keywords each chunk matches."""

import logging
import re
from typing import Any, Dict, List, Set

from app.database.sqlite_db import SQLiteDB
from app.utils.config import settings

logger = logging.getLogger(__name__)

# common english stop words to filter out from queries
STOP_WORDS = {
    "the", "is", "a", "an", "of", "to", "in", "for", "and",
    "how", "what", "where", "when", "why", "who", "which",
    "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by",
    "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
}


def _extract_keywords(query: str) -> Set[str]:
    """Lowercase, tokenize, remove stop words, return unique terms."""
    query = query.lower()
    tokens = re.split(r'[^a-z0-9]+', query)
    return {t for t in tokens if t and t not in STOP_WORDS}


def _highlight_and_score(text: str, query_keywords: Set[str]):
    """
    Compute keyword match % and wrap matched words in <mark> tags.
    Returns (score, matched_list, unmatched_list, highlighted_text)
    """
    if not query_keywords:
        return 0.0, [], [], text

    matched = set()
    unmatched = set(query_keywords)

    highlighted_text = text
    for kw in query_keywords:
        pattern = re.compile(rf'\b({re.escape(kw)})\b', flags=re.IGNORECASE)
        if pattern.search(highlighted_text):
            matched.add(kw)
            unmatched.discard(kw)
            highlighted_text = pattern.sub(r'<mark>\1</mark>', highlighted_text)

    score = (len(matched) / len(query_keywords)) * 100.0

    # truncate really long texts for display
    words = highlighted_text.split()
    if len(words) > settings.SNIPPET_LENGTH * 2:
        highlighted_text = " ".join(words[: settings.SNIPPET_LENGTH * 2]) + "..."
        # make sure we don't leave an unclosed <mark> tag
        if highlighted_text.count("<mark>") > highlighted_text.count("</mark>"):
            highlighted_text += "</mark>"

    return score, sorted(list(matched)), sorted(list(unmatched)), highlighted_text


class KeywordSearchService:
    """Search chunks by keyword match percentage."""

    def __init__(self, db: SQLiteDB):
        self._db = db

    def search(self, query: str, top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        """
        Search indexed chunks and score them by % of query keywords matched.
        """
        keywords = _extract_keywords(query)

        if not keywords:
            logger.debug("KeywordSearch: query=%r has no meaningful keywords", query)
            return []

        # build FTS query: "word1" OR "word2" etc
        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)

        # grab extra candidates so we can re-score them properly
        candidates = self._db.keyword_search(query=fts_query, top_k=max(top_k * 5, 50))

        results = []
        for c in candidates:
            score, matched, unmatched, snippet = _highlight_and_score(c["text"], keywords)
            if score > 0:
                c["keyword_score"] = score
                c["matched_keywords"] = matched
                c["unmatched_keywords"] = unmatched
                c["text"] = snippet
                c.pop("raw_rank", None)
                results.append(c)

        # sort by score, highest first
        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        results = results[:top_k]

        logger.debug("KeywordSearch: query=%r → %d results", query, len(results))
        return results

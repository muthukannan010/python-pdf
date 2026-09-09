"""Keyword search - computes what % of query keywords each chunk matches."""

import logging
import re
import unicodedata
from collections import defaultdict
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


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)


def _normalize_token(token: str) -> str:
    """Normalize tokens consistently with the text stored in SQLite."""
    token = unicodedata.normalize("NFKC", token).casefold()
    return token.replace("’", "'").strip("'")


def _stem_token(token: str) -> str:
    """Apply conservative stemming compatible with common Porter results."""
    token = _normalize_token(token)
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def _extract_keywords(query: str) -> Set[str]:
    """Tokenize query text, normalize it, and remove stop words."""
    return {
        token
        for raw_token in _TOKEN_RE.findall(query)
        if (token := _normalize_token(raw_token)) and token not in STOP_WORDS
    }


def _highlight_and_score(text: str, query_keywords: Set[str]):
    """
    Compute keyword match % and wrap matched words in <mark> tags.
    Returns (score, matched_list, unmatched_list, highlighted_text)
    """
    if not query_keywords:
        return 0.0, [], [], text

    query_by_stem = defaultdict(set)
    for keyword in query_keywords:
        query_by_stem[_stem_token(keyword)].add(keyword)

    matched = set()
    token_spans = []
    for match in _TOKEN_RE.finditer(text):
        token = _normalize_token(match.group())
        matching_keywords = query_by_stem.get(_stem_token(token), set())
        if matching_keywords:
            matched.update(matching_keywords)
            token_spans.append((match.start(), match.end()))

    highlighted_text = text
    for start, end in reversed(token_spans):
        highlighted_text = (
            highlighted_text[:start]
            + "<mark>"
            + highlighted_text[start:end]
            + "</mark>"
            + highlighted_text[end:]
        )

    score = (len(matched) / len(query_keywords)) * 100.0
    unmatched = query_keywords - matched

    # truncate really long texts for display
    words = text.split()
    if len(words) > settings.SNIPPET_LENGTH * 2:
        display_text = " ".join(words[: settings.SNIPPET_LENGTH * 2])
        display_spans = []
        for match in _TOKEN_RE.finditer(display_text):
            if _stem_token(_normalize_token(match.group())) in query_by_stem:
                display_spans.append((match.start(), match.end()))
        highlighted_text = display_text
        for start, end in reversed(display_spans):
            highlighted_text = (
                highlighted_text[:start]
                + "<mark>"
                + highlighted_text[start:end]
                + "</mark>"
                + highlighted_text[end:]
            )
        highlighted_text += "..."

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

        query_stems = {_stem_token(keyword) for keyword in keywords}

        # build FTS query: "word1" OR "word2" etc
        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)

        # grab extra candidates so we can re-score them properly
        candidates = self._db.keyword_search(query=fts_query, top_k=max(top_k * 5, 50))

        query_by_stem = defaultdict(set)
        for keyword in keywords:
            query_by_stem[_stem_token(keyword)].add(keyword)

        results = []
        for c in candidates:
            score, matched, unmatched, snippet = _highlight_and_score(c["text"], keywords)
            if score > 0:
                c["keyword_score"] = score
                c["matched_keywords"] = matched
                c["unmatched_keywords"] = unmatched
                c["text"] = snippet
                c.pop("raw_rank", None)
                
                # filter locations to only include matched keywords
                if "locations" in c:
                    filtered_locs = []
                    for loc in c["locations"]:
                        loc_text = loc.get("text", "")
                        if any(
                            _stem_token(_normalize_token(token))
                            in query_by_stem
                            for token in _TOKEN_RE.findall(loc_text)
                        ):
                            filtered_locs.append(loc)
                    c["locations"] = filtered_locs
                    
                results.append(c)

        # sort by score, highest first
        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        results = results[:top_k]

        logger.debug("KeywordSearch: query=%r → %d results", query, len(results))
        return results

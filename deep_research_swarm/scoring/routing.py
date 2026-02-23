"""Backend routing — deterministic query classification and backend selection (V7, PR-07).

Classifies sub-queries by type (academic, general, archival, technical) using
keyword signal density. Routes each type to appropriate backends. No LLM.
Microsecond speed.
"""

from __future__ import annotations

import re
from enum import Enum

# Year regex for archival detection (year < current_year - 5)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_ARXIV_RE = re.compile(r"\d{4}\.\d{4,}")
_URL_RE = re.compile(r"https?://\S+")

# Current year for archival heuristic (5-year lookback)
_ARCHIVAL_YEAR_THRESHOLD = 2021


class QueryType(str, Enum):
    ACADEMIC = "academic"
    GENERAL = "general"
    ARCHIVAL = "archival"
    TECHNICAL = "technical"


BACKEND_ROUTES: dict[QueryType, list[str]] = {
    QueryType.ACADEMIC: ["openalex", "semantic_scholar", "searxng"],
    QueryType.GENERAL: ["searxng", "exa", "tavily"],
    QueryType.ARCHIVAL: ["wayback", "searxng"],
    QueryType.TECHNICAL: ["searxng", "exa"],
}

_ACADEMIC_KEYWORDS: set[str] = {
    "paper",
    "study",
    "research",
    "journal",
    "published",
    "cited",
    "peer-reviewed",
    "meta-analysis",
    "systematic review",
    "methodology",
    "hypothesis",
    "findings",
    "literature",
    "scholars",
    "academic",
    "doi:",
    "arxiv:",
    "et al.",
}

_ARCHIVAL_KEYWORDS: set[str] = {
    "historical",
    "archive",
    "was",
    "used to be",
    "original version",
    "removed",
    "deleted",
    "old version",
    "wayback",
    "cached",
    "snapshot",
    "no longer available",
    "previously",
}

_TECHNICAL_KEYWORDS: set[str] = {
    "documentation",
    "api",
    "library",
    "framework",
    "implementation",
    "code",
    "tutorial",
    "github",
    "npm",
    "pip",
    "how to",
    "error",
    "debug",
    "stack overflow",
    "package",
}

# Tie-breaker priority: ACADEMIC > TECHNICAL > ARCHIVAL > GENERAL
_TIEBREAKER = {
    QueryType.ACADEMIC: 0,
    QueryType.TECHNICAL: 1,
    QueryType.ARCHIVAL: 2,
    QueryType.GENERAL: 3,
}

# General fallback chain when no preferred backends available
_GENERAL_FALLBACK = ["searxng", "exa", "tavily"]


# Multi-word keywords that need substring matching (not word boundary)
_MULTI_WORD_KEYWORDS = {
    "used to be",
    "original version",
    "old version",
    "no longer available",
    "systematic review",
    "meta-analysis",
    "peer-reviewed",
    "et al.",
    "how to",
    "stack overflow",
}

# Single-word keywords that need word boundary matching
_SINGLE_WORD_ACADEMIC = _ACADEMIC_KEYWORDS - _MULTI_WORD_KEYWORDS
_SINGLE_WORD_ARCHIVAL = _ARCHIVAL_KEYWORDS - _MULTI_WORD_KEYWORDS
_SINGLE_WORD_TECHNICAL = _TECHNICAL_KEYWORDS - _MULTI_WORD_KEYWORDS


def _count_keyword_matches(query_lower: str, keywords: set[str]) -> int:
    """Count how many keyword signals appear in the query.

    Uses word-boundary matching for single-word keywords to avoid
    false positives (e.g. 'restaurants' matching 'research').
    Multi-word phrases use substring matching.
    """
    count = 0
    query_words = set(query_lower.split())
    for kw in keywords:
        if " " in kw or "." in kw or ":" in kw:
            # Multi-word/punctuated: substring match
            if kw in query_lower:
                count += 1
        else:
            # Single word: word boundary match
            if kw in query_words:
                count += 1
    return count


def classify_query(query: str) -> QueryType:
    """Classify a sub-query by type using keyword signal density.

    Returns the dominant type based on signal count.
    Tie-breaker: ACADEMIC > TECHNICAL > ARCHIVAL > GENERAL.
    Short queries (< 4 words) bias toward GENERAL.
    URL-containing queries bias toward ARCHIVAL.
    """
    query_lower = query.lower()
    words = query.split()

    scores: dict[QueryType, int] = {
        QueryType.ACADEMIC: 0,
        QueryType.GENERAL: 0,
        QueryType.ARCHIVAL: 0,
        QueryType.TECHNICAL: 0,
    }

    # Keyword matches
    scores[QueryType.ACADEMIC] += _count_keyword_matches(query_lower, _ACADEMIC_KEYWORDS)
    scores[QueryType.ARCHIVAL] += _count_keyword_matches(query_lower, _ARCHIVAL_KEYWORDS)
    scores[QueryType.TECHNICAL] += _count_keyword_matches(query_lower, _TECHNICAL_KEYWORDS)

    # Regex signals for ACADEMIC
    if _DOI_RE.search(query):
        scores[QueryType.ACADEMIC] += 2
    if _ARXIV_RE.search(query):
        scores[QueryType.ACADEMIC] += 2

    # Year signal for ARCHIVAL
    for match in _YEAR_RE.finditer(query):
        year = int(match.group(1))
        if year < _ARCHIVAL_YEAR_THRESHOLD:
            scores[QueryType.ARCHIVAL] += 1

    # URL presence biases toward ARCHIVAL
    if _URL_RE.search(query):
        scores[QueryType.ARCHIVAL] += 2

    # Very short queries bias toward GENERAL
    if len(words) < 4:
        scores[QueryType.GENERAL] += 2

    # Find dominant type (highest score, tiebreaker by priority)
    # If all scores are 0, default to GENERAL
    max_score = max(scores.values())
    if max_score == 0:
        return QueryType.GENERAL

    best_type = QueryType.GENERAL
    best_score = 0

    for qt, sc in scores.items():
        if sc > best_score or (sc == best_score and _TIEBREAKER[qt] < _TIEBREAKER[best_type]):
            best_type = qt
            best_score = sc

    return best_type


def route_backends(query_type: QueryType, available: list[str]) -> list[str]:
    """Return ordered backend list for query type, filtered to available backends.

    If no preferred backends are available, falls back to first available
    from the general fallback chain. Always returns at least one backend.
    """
    preferred = BACKEND_ROUTES.get(query_type, BACKEND_ROUTES[QueryType.GENERAL])
    filtered = [b for b in preferred if b in available]

    if filtered:
        return filtered

    # Fallback: first available from general chain
    for fallback in _GENERAL_FALLBACK:
        if fallback in available:
            return [fallback]

    # Last resort: return first available backend
    if available:
        return [available[0]]

    return ["searxng"]

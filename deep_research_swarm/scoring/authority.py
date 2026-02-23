"""Source authority classification and scoring."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from deep_research_swarm.contracts import SourceAuthority

# TLD and domain patterns for classification
_INSTITUTIONAL_TLDS = {".edu", ".gov", ".mil"}
_INSTITUTIONAL_DOMAINS = {
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "jstor.org",
    "ieee.org",
    "acm.org",
    "nih.gov",
    "who.int",
    "un.org",
    "worldbank.org",
}

# V7: Scholarly-specific domains (always institutional)
_SCHOLARLY_DOMAINS = {
    "doi.org",
    "openalex.org",
    "semanticscholar.org",
    "crossref.org",
    "orcid.org",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "plos.org",
    "plosone.org",
    "frontiersin.org",
    "mdpi.com",
    "tandfonline.com",
    "sagepub.com",
    "cambridge.org",
    "oxfordjournals.org",
    "academic.oup.com",
    "europepmc.org",
    "biomedcentral.com",
    "researchgate.net",
}

_PROFESSIONAL_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "techcrunch.com",
    "arstechnica.com",
    "wired.com",
    "github.com",
    "stackoverflow.com",
    "medium.com",
    "substack.com",
}

_COMMUNITY_DOMAINS = {
    "reddit.com",
    "wikipedia.org",
    "wikimedia.org",
    "quora.com",
    "stackexchange.com",
    "discord.com",
    "fandom.com",
}

_PROMOTIONAL_PATTERNS = {
    "shop.",
    "store.",
    "buy.",
    "deals.",
    "promo.",
}

# Authority -> numeric score mapping
_AUTHORITY_SCORES: dict[SourceAuthority, float] = {
    SourceAuthority.INSTITUTIONAL: 0.95,
    SourceAuthority.PROFESSIONAL: 0.75,
    SourceAuthority.COMMUNITY: 0.50,
    SourceAuthority.PROMOTIONAL: 0.15,
    SourceAuthority.UNKNOWN: 0.40,
}


def classify_authority(url: str) -> SourceAuthority:
    """Classify a URL's source authority level."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return SourceAuthority.UNKNOWN

    if not hostname:
        return SourceAuthority.UNKNOWN

    # Strip www.
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Check TLDs
    for tld in _INSTITUTIONAL_TLDS:
        if hostname.endswith(tld):
            return SourceAuthority.INSTITUTIONAL

    # Check exact domain matches
    # Also check parent domain (e.g., sub.nature.com -> nature.com)
    parts = hostname.split(".")
    for i in range(len(parts) - 1):
        domain = ".".join(parts[i:])
        if domain in _INSTITUTIONAL_DOMAINS:
            return SourceAuthority.INSTITUTIONAL
        if domain in _SCHOLARLY_DOMAINS:
            return SourceAuthority.INSTITUTIONAL
        if domain in _PROFESSIONAL_DOMAINS:
            return SourceAuthority.PROFESSIONAL
        if domain in _COMMUNITY_DOMAINS:
            return SourceAuthority.COMMUNITY

    # Check promotional patterns
    for pattern in _PROMOTIONAL_PATTERNS:
        if hostname.startswith(pattern):
            return SourceAuthority.PROMOTIONAL

    return SourceAuthority.UNKNOWN


def authority_score(authority: SourceAuthority) -> float:
    """Convert authority classification to numeric score."""
    return _AUTHORITY_SCORES.get(authority, 0.40)


# --- V7: Enhanced authority scoring with scholarly metadata (PR-09) ---

# Citation count -> bonus mapping (log-scale tiers)
_CITATION_BONUS: list[tuple[int, float]] = [
    (500, 0.10),  # 500+ citations: strong signal
    (100, 0.07),
    (50, 0.05),
    (10, 0.03),
    (1, 0.01),
]


def _citation_bonus(citation_count: int) -> float:
    """Compute citation-based bonus for authority score."""
    for threshold, bonus in _CITATION_BONUS:
        if citation_count >= threshold:
            return bonus
    return 0.0


def score_authority(
    url: str,
    *,
    scholarly_metadata: dict[str, Any] | None = None,
) -> tuple[SourceAuthority, float]:
    """Enhanced authority scoring that uses scholarly metadata when available.

    Returns (authority_classification, numeric_score).

    When scholarly_metadata is present:
    - Base score comes from URL classification (same as before)
    - Citation count adds a bonus (capped at 1.0)
    - Open access gives a small trustworthiness boost
    - Peer-reviewed venue gets a venue bonus

    When scholarly_metadata is absent, falls back to classify_authority()
    + authority_score() (identical to V6 behavior).
    """
    classification = classify_authority(url)
    base_score = authority_score(classification)

    if not scholarly_metadata:
        return (classification, base_score)

    # With metadata: always at least INSTITUTIONAL
    if classification not in (
        SourceAuthority.INSTITUTIONAL,
        SourceAuthority.PROFESSIONAL,
    ):
        classification = SourceAuthority.INSTITUTIONAL
        base_score = authority_score(classification)

    # Citation bonus
    cite_count = scholarly_metadata.get("citation_count", 0) or 0
    bonus = _citation_bonus(cite_count)

    # Open access bonus (small: trustworthiness signal, not quality)
    if scholarly_metadata.get("is_open_access", False):
        bonus += 0.02

    # Venue bonus: having a venue at all indicates peer review
    venue = scholarly_metadata.get("venue", "")
    if venue:
        bonus += 0.02

    return (classification, round(min(1.0, base_score + bonus), 4))

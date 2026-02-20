"""Source authority classification and scoring."""

from __future__ import annotations

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

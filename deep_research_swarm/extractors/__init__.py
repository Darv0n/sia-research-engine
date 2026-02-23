"""Content extraction cascade."""

from __future__ import annotations

from urllib.parse import urlparse

from .crawl4ai_extractor import extract_with_crawl4ai
from .pdf_extractor import extract_pdf
from .trafilatura_extractor import extract_with_trafilatura


def _is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF based on path extension."""
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def _is_wayback_url(url: str) -> bool:
    """Check if URL is already a Wayback Machine archive URL."""
    return "web.archive.org" in url


async def _try_wayback_fallback(url: str) -> tuple[str, str]:
    """Attempt Wayback Machine fallback for a URL (V7, PR-05).

    Only called when all live extractors fail. Looks up the most
    recent snapshot and extracts content from it.
    """
    try:
        from deep_research_swarm.backends.wayback import WaybackBackend

        wayback = WaybackBackend()
        capture = await wayback.lookup_url(url)
        if not capture:
            return "", "none"

        content = await wayback.fetch_archived_content(capture)
        if content:
            # Run the archived HTML through trafilatura for clean extraction
            from trafilatura import extract as traf_extract

            cleaned = traf_extract(content)
            if cleaned:
                return cleaned, "wayback+trafilatura"
            # Fall back to raw content if trafilatura can't parse
            return content[:50000], "wayback"
    except Exception:
        pass
    return "", "none"


async def extract(url: str, *, wayback_fallback: bool = True) -> tuple[str, str]:
    """Extract content from URL using cascade: PDF -> Crawl4AI -> Trafilatura -> Wayback.

    Returns (content, extractor_name) tuple.
    Wayback fallback only triggered when all live extractors fail (D6).
    """
    if _is_pdf_url(url):
        content = await extract_pdf(url)
        if content:
            return content, "pdf"

    content = await extract_with_crawl4ai(url)
    if content:
        return content, "crawl4ai"

    content = await extract_with_trafilatura(url)
    if content:
        return content, "trafilatura"

    # V7: Wayback fallback — only for non-archive URLs when all live extractors fail
    if wayback_fallback and not _is_wayback_url(url):
        content, extractor = await _try_wayback_fallback(url)
        if content:
            return content, extractor

    return "", "none"

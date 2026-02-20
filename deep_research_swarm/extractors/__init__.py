"""Content extraction cascade."""

from __future__ import annotations

from .crawl4ai_extractor import extract_with_crawl4ai
from .trafilatura_extractor import extract_with_trafilatura


async def extract(url: str) -> tuple[str, str]:
    """Extract content from URL using cascade: Crawl4AI -> Trafilatura.

    Returns (content, extractor_name) tuple.
    """
    content = await extract_with_crawl4ai(url)
    if content:
        return content, "crawl4ai"

    content = await extract_with_trafilatura(url)
    if content:
        return content, "trafilatura"

    return "", "none"

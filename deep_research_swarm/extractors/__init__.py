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


async def extract(url: str) -> tuple[str, str]:
    """Extract content from URL using cascade: PDF -> Crawl4AI -> Trafilatura.

    Returns (content, extractor_name) tuple.
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

    return "", "none"

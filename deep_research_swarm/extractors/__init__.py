"""Content extraction cascade (I8).

V8 cascade: PDF (pymupdf4llm) -> GROBID -> OCR -> Crawl4AI -> Trafilatura -> Wayback
"""

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


async def _try_grobid_fallback(url: str, grobid_url: str) -> tuple[str, str]:
    """Attempt GROBID extraction for a PDF URL (V8, I8).

    Returns (content, "grobid") on success, ("", "none") on failure.
    GROBID also returns references, but we only use content in the cascade.
    """
    if not grobid_url:
        return "", "none"
    try:
        from .grobid_extractor import extract_with_grobid

        content, _refs = await extract_with_grobid(url, grobid_url=grobid_url)
        if content and len(content.strip()) > 50:
            return content.strip(), "grobid"
    except Exception:
        pass
    return "", "none"


async def _try_ocr_fallback(url: str) -> tuple[str, str]:
    """Attempt OCR extraction for a scanned PDF (V8, I8).

    Returns (content, "ocr") on success, ("", "none") on failure.
    """
    try:
        from .ocr_extractor import extract_with_ocr

        content = await extract_with_ocr(url)
        if content and len(content.strip()) > 50:
            return content.strip(), "ocr"
    except Exception:
        pass
    return "", "none"


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


async def extract(
    url: str,
    *,
    wayback_fallback: bool = True,
    grobid_url: str = "",
) -> tuple[str, str]:
    """Extract content using cascade: PDF -> GROBID -> OCR -> Crawl4AI -> Trafilatura -> Wayback.

    Returns (content, extractor_name) tuple.
    I8: GROBID and OCR extend the cascade, do not replace it.
    Wayback fallback only triggered when all live extractors fail (D6).
    """
    is_pdf = _is_pdf_url(url)

    if is_pdf:
        # Stage 1: PyMuPDF4LLM (fast, local)
        content = await extract_pdf(url)
        if content:
            return content, "pdf"

        # Stage 2: GROBID (structured, needs server)
        if grobid_url:
            content, extractor = await _try_grobid_fallback(url, grobid_url)
            if content:
                return content, extractor

        # Stage 3: OCR (scanned PDFs, needs paddleocr)
        content, extractor = await _try_ocr_fallback(url)
        if content:
            return content, extractor

    # Stage 4: Crawl4AI (web pages)
    content = await extract_with_crawl4ai(url)
    if content:
        return content, "crawl4ai"

    # Stage 5: Trafilatura (web pages)
    content = await extract_with_trafilatura(url)
    if content:
        return content, "trafilatura"

    # Stage 6: Wayback fallback — only for non-archive URLs when all live extractors fail
    if wayback_fallback and not _is_wayback_url(url):
        content, extractor = await _try_wayback_fallback(url)
        if content:
            return content, extractor

    return "", "none"

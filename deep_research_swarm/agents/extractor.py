"""Extractor agent — coordinates content extraction from search results."""

from __future__ import annotations

import uuid

from deep_research_swarm.contracts import ExtractedContent, SearchResult
from deep_research_swarm.extractors import extract


async def extract_content(
    search_result: SearchResult,
    *,
    content_truncation_chars: int = 50000,
    grobid_url: str = "",
) -> ExtractedContent:
    """Extract content from a single search result URL.

    Uses the extraction cascade: PDF -> GROBID -> OCR -> Crawl4AI -> Trafilatura -> Wayback.
    """
    url = search_result["url"]

    try:
        content, extractor_used = await extract(url, grobid_url=grobid_url)
        success = bool(content)
        error = None if success else "No content extracted"
    except Exception as e:
        content = ""
        extractor_used = "none"
        success = False
        error = str(e)

    return ExtractedContent(
        id=f"ec-{uuid.uuid4().hex[:8]}",
        search_result_id=search_result["id"],
        url=url,
        title=search_result["title"],
        content=content[:content_truncation_chars],
        content_length=len(content),
        extractor_used=extractor_used,
        extraction_success=success,
        error=error,
    )

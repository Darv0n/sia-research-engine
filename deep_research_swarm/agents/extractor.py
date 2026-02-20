"""Extractor agent — coordinates content extraction from search results."""

from __future__ import annotations

import uuid

from deep_research_swarm.contracts import ExtractedContent, SearchResult
from deep_research_swarm.extractors import extract


async def extract_content(search_result: SearchResult) -> ExtractedContent:
    """Extract content from a single search result URL.

    Uses the extraction cascade: Crawl4AI -> Trafilatura.
    """
    url = search_result["url"]

    try:
        content, extractor_used = await extract(url)
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
        content=content[:50000],  # Cap at 50k chars per document
        content_length=len(content),
        extractor_used=extractor_used,
        extraction_success=success,
        error=error,
    )

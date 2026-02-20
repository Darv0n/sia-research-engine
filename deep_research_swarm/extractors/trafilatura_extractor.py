"""Trafilatura extractor — fast static HTML extraction fallback."""

from __future__ import annotations


async def extract_with_trafilatura(url: str) -> str:
    """Extract content from URL using Trafilatura.

    Returns cleaned markdown content, or empty string on failure.
    """
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""

        content = trafilatura.extract(
            downloaded,
            output_format="txt",
            include_links=False,
            include_tables=True,
        )

        if content and len(content.strip()) > 100:
            return content.strip()

        return ""
    except Exception:
        return ""

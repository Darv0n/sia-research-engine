"""PDF extractor — PyMuPDF4LLM for PDF URLs (V4)."""

from __future__ import annotations


async def extract_pdf(path_or_url: str) -> str:
    """Extract content from a PDF file using PyMuPDF4LLM.

    Returns markdown content, or empty string on failure.
    Stub for V4 implementation.
    """
    try:
        import pymupdf4llm

        content = pymupdf4llm.to_markdown(path_or_url)
        if content and len(content.strip()) > 50:
            return content.strip()
        return ""
    except Exception:
        return ""

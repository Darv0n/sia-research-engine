"""PDF extractor — PyMuPDF4LLM for PDF URLs and local files (V4)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx
import pymupdf4llm


async def extract_pdf(path_or_url: str) -> str:
    """Extract content from a PDF file or URL using PyMuPDF4LLM.

    For HTTP(S) URLs: downloads to a temp file, extracts, cleans up.
    For local paths: extracts directly.

    Returns markdown content, or empty string on failure.
    """
    if path_or_url.startswith(("http://", "https://")):
        return await _extract_from_url(path_or_url)
    return await _extract_from_path(path_or_url)


async def _extract_from_url(url: str) -> str:
    """Download PDF from URL to temp file and extract."""
    tmp_path: str | None = None
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        Path(tmp_path).write_bytes(resp.content)

        return await _extract_from_path(tmp_path)
    except Exception:
        return ""
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()


async def _extract_from_path(path: str) -> str:
    """Extract markdown from a local PDF file (CPU-bound, runs in thread)."""
    try:
        content = await asyncio.to_thread(pymupdf4llm.to_markdown, path)
        if content and len(content.strip()) > 50:
            return content.strip()
        return ""
    except Exception:
        return ""

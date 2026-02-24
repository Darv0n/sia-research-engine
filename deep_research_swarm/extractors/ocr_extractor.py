"""OCR extractor — PaddleOCR fallback for scanned/image PDFs (V8, I8).

PaddleOCR (Apache-2.0) provides OCR for PDFs that GROBID and PyMuPDF
can't parse (pure image scans). Optional dependency via pip install .[ocr].

Falls back gracefully when paddleocr is not installed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx


async def extract_with_ocr(
    path_or_url: str,
) -> str:
    """Extract text from a scanned PDF or image using PaddleOCR.

    Args:
        path_or_url: PDF/image file path or URL.

    Returns extracted text, or empty string on failure or if paddleocr unavailable.
    """
    import importlib.util

    if importlib.util.find_spec("paddleocr") is None:
        return ""

    pdf_path: str | None = None
    tmp_path: str | None = None

    try:
        if path_or_url.startswith(("http://", "https://")):
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(path_or_url)
                resp.raise_for_status()

            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            Path(tmp_path).write_bytes(resp.content)
            pdf_path = tmp_path
        else:
            pdf_path = path_or_url

        # PaddleOCR is CPU-bound — run in thread
        text = await asyncio.to_thread(_run_ocr, pdf_path)
        return text

    except Exception:
        return ""
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()


def _run_ocr(path: str) -> str:
    """Run PaddleOCR on a file (synchronous, CPU-bound)."""
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(path, cls=True)

        if not result:
            return ""

        lines: list[str] = []
        for page in result:
            if not page:
                continue
            for line in page:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    if text.strip():
                        lines.append(text.strip())

        return "\n".join(lines)

    except Exception:
        return ""

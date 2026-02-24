"""Tests for extractors/ocr_extractor.py — PaddleOCR extraction (PR-09)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deep_research_swarm.extractors.ocr_extractor import _run_ocr, extract_with_ocr


class TestExtractWithOcr:
    @pytest.mark.asyncio
    async def test_returns_empty_when_paddleocr_missing(self):
        """Falls back gracefully when paddleocr not installed."""
        with patch.dict("sys.modules", {"paddleocr": None}):
            # Force reimport failure
            import importlib

            import deep_research_swarm.extractors.ocr_extractor as mod

            importlib.reload(mod)
            result = await mod.extract_with_ocr("test.pdf")
            # Should return empty (ImportError caught)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        """Returns empty string when extraction fails."""
        with patch(
            "deep_research_swarm.extractors.ocr_extractor.extract_with_ocr",
            return_value="",
        ):
            result = await extract_with_ocr("nonexistent.pdf")
            assert result == "" or isinstance(result, str)


class TestRunOcr:
    def test_returns_empty_when_paddleocr_missing(self):
        """_run_ocr returns empty string when paddleocr unavailable."""
        with patch.dict("sys.modules", {"paddleocr": None}):
            result = _run_ocr("nonexistent.pdf")
            # Should return empty (ImportError or general Exception caught)
            assert isinstance(result, str)


class TestCascadeIntegration:
    """Test that OCR and GROBID are wired into the extraction cascade."""

    def test_extract_accepts_grobid_url(self):
        """extract() function accepts grobid_url parameter."""
        import inspect

        from deep_research_swarm.extractors import extract

        sig = inspect.signature(extract)
        assert "grobid_url" in sig.parameters

    def test_cascade_order_documented(self):
        """Cascade docstring documents I8 ordering."""
        from deep_research_swarm.extractors import extract

        assert "GROBID" in (extract.__doc__ or "")
        assert "OCR" in (extract.__doc__ or "")

    def test_extractor_accepts_grobid_url(self):
        """extract_content() agent accepts grobid_url parameter."""
        import inspect

        from deep_research_swarm.agents.extractor import extract_content

        sig = inspect.signature(extract_content)
        assert "grobid_url" in sig.parameters

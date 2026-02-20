"""Tests for V4 PDF extraction — URL detection, cascade routing, extractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research_swarm.extractors import _is_pdf_url


class TestPdfUrlDetection:
    def test_pdf_extension(self):
        assert _is_pdf_url("https://example.com/paper.pdf") is True

    def test_pdf_extension_uppercase(self):
        assert _is_pdf_url("https://example.com/paper.PDF") is True

    def test_html_extension(self):
        assert _is_pdf_url("https://example.com/page.html") is False

    def test_no_extension(self):
        assert _is_pdf_url("https://example.com/page") is False

    def test_pdf_in_path_not_extension(self):
        assert _is_pdf_url("https://example.com/pdf/page.html") is False

    def test_pdf_with_query_params(self):
        """URL with .pdf path and query params is still detected as PDF."""
        assert _is_pdf_url("https://example.com/paper.pdf?dl=1") is True

    def test_pdf_path_only(self):
        """Pure .pdf path without query string."""
        assert _is_pdf_url("https://arxiv.org/pdf/2301.00001.pdf") is True


class TestCascadeRouting:
    @pytest.mark.asyncio
    async def test_pdf_url_routes_to_pdf_extractor(self):
        """PDF URLs should try PDF extractor first."""
        from deep_research_swarm.extractors import extract

        with patch(
            "deep_research_swarm.extractors.extract_pdf",
            new_callable=AsyncMock,
            return_value="# PDF Content",
        ) as mock_pdf:
            content, extractor = await extract("https://example.com/paper.pdf")
            assert extractor == "pdf"
            assert content == "# PDF Content"
            mock_pdf.assert_called_once_with("https://example.com/paper.pdf")

    @pytest.mark.asyncio
    async def test_pdf_failure_falls_through(self):
        """If PDF extraction fails, cascade continues to crawl4ai."""
        from deep_research_swarm.extractors import extract

        with (
            patch(
                "deep_research_swarm.extractors.extract_pdf",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "deep_research_swarm.extractors.extract_with_crawl4ai",
                new_callable=AsyncMock,
                return_value="HTML content",
            ),
        ):
            content, extractor = await extract("https://example.com/paper.pdf")
            assert extractor == "crawl4ai"

    @pytest.mark.asyncio
    async def test_non_pdf_skips_pdf_extractor(self):
        """Non-PDF URLs should skip PDF extractor entirely."""
        from deep_research_swarm.extractors import extract

        with (
            patch(
                "deep_research_swarm.extractors.extract_pdf",
                new_callable=AsyncMock,
            ) as mock_pdf,
            patch(
                "deep_research_swarm.extractors.extract_with_crawl4ai",
                new_callable=AsyncMock,
                return_value="HTML content",
            ),
        ):
            content, extractor = await extract("https://example.com/page.html")
            mock_pdf.assert_not_called()
            assert extractor == "crawl4ai"


class TestPdfExtractor:
    @pytest.mark.asyncio
    async def test_extract_from_local_path(self, tmp_path):
        """Local PDF extraction via pymupdf4llm.to_markdown."""
        from deep_research_swarm.extractors.pdf_extractor import extract_pdf

        with patch("deep_research_swarm.extractors.pdf_extractor.pymupdf4llm") as mock_pymupdf:
            mock_pymupdf.to_markdown.return_value = "# Extracted\n" + "content " * 20
            result = await extract_pdf(str(tmp_path / "test.pdf"))
            mock_pymupdf.to_markdown.assert_called_once()
            assert "Extracted" in result

    @pytest.mark.asyncio
    async def test_extract_from_url_downloads(self):
        """HTTP URLs are downloaded before extraction."""
        from deep_research_swarm.extractors.pdf_extractor import extract_pdf

        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 fake content"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "deep_research_swarm.extractors.pdf_extractor.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch("deep_research_swarm.extractors.pdf_extractor.pymupdf4llm") as mock_pymupdf,
        ):
            mock_pymupdf.to_markdown.return_value = "# From URL\n" + "content " * 20
            result = await extract_pdf("https://example.com/paper.pdf")
            assert "From URL" in result
            mock_client.get.assert_called_once_with("https://example.com/paper.pdf")

    @pytest.mark.asyncio
    async def test_extract_failure_returns_empty(self):
        """Extraction failure returns empty string."""
        from deep_research_swarm.extractors.pdf_extractor import extract_pdf

        with patch("deep_research_swarm.extractors.pdf_extractor.pymupdf4llm") as mock_pymupdf:
            mock_pymupdf.to_markdown.side_effect = RuntimeError("corrupt PDF")
            result = await extract_pdf("/fake/path.pdf")
            assert result == ""

    @pytest.mark.asyncio
    async def test_short_content_returns_empty(self):
        """Content under 50 chars is treated as empty."""
        from deep_research_swarm.extractors.pdf_extractor import extract_pdf

        with patch("deep_research_swarm.extractors.pdf_extractor.pymupdf4llm") as mock_pymupdf:
            mock_pymupdf.to_markdown.return_value = "Short"
            result = await extract_pdf("/fake/path.pdf")
            assert result == ""

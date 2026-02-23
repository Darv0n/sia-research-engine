"""Tests for backends/crossref.py — DOI resolution utilities (PR-04)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research_swarm.backends.crossref import (
    _crossref_to_scholarly,
    enrich_scholarly_result,
    extract_doi_from_url,
    normalize_doi,
)
from deep_research_swarm.contracts import SearchResult, SourceAuthority


class TestNormalizeDoi:
    def test_bare_doi(self):
        assert normalize_doi("10.1038/s41586-023-06768-0") == "10.1038/s41586-023-06768-0"

    def test_https_doi_org(self):
        assert normalize_doi("https://doi.org/10.1038/test") == "10.1038/test"

    def test_http_dx_doi_org(self):
        assert normalize_doi("http://dx.doi.org/10.1038/test") == "10.1038/test"

    def test_doi_prefix(self):
        assert normalize_doi("doi:10.1038/test") == "10.1038/test"

    def test_whitespace_stripped(self):
        assert normalize_doi("  10.1038/test  ") == "10.1038/test"


class TestExtractDoiFromUrl:
    def test_doi_url(self):
        assert extract_doi_from_url("https://doi.org/10.1038/s41586-023-06768-0") == (
            "10.1038/s41586-023-06768-0"
        )

    def test_embedded_doi(self):
        url = "https://www.nature.com/articles/10.1038/s41586-023-06768-0"
        doi = extract_doi_from_url(url)
        assert doi == "10.1038/s41586-023-06768-0"

    def test_no_doi(self):
        assert extract_doi_from_url("https://example.com/page") is None

    def test_arxiv_no_doi(self):
        assert extract_doi_from_url("https://arxiv.org/abs/2401.12345") is None


class TestCrossrefToScholarly:
    def test_full_work(self):
        work = {
            "DOI": "10.1038/test",
            "title": ["Quantum Entanglement"],
            "author": [
                {"given": "Alice", "family": "Smith"},
                {"given": "Bob", "family": "Jones"},
            ],
            "published-print": {"date-parts": [[2023, 6]]},
            "container-title": ["Nature"],
            "is-referenced-by-count": 100,
            "references-count": 30,
            "abstract": "An abstract about entanglement.",
            "license": [],
        }
        sm = _crossref_to_scholarly(work)
        assert sm["doi"] == "10.1038/test"
        assert sm["year"] == 2023
        assert sm["venue"] == "Nature"
        assert sm["citation_count"] == 100
        assert sm["reference_count"] == 30
        assert len(sm["authors"]) == 2

    def test_minimal_work(self):
        sm = _crossref_to_scholarly({})
        assert sm["doi"] == ""
        assert sm["year"] == 0
        assert sm["authors"] == []

    def test_cc_license_detected_as_oa(self):
        work = {
            "DOI": "10.1234/test",
            "title": ["Test"],
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        }
        sm = _crossref_to_scholarly(work)
        assert sm["is_open_access"] is True


class TestEnrichScholarlyResult:
    @pytest.mark.asyncio
    async def test_skip_if_already_enriched(self):
        sr = SearchResult(
            id="sr-test",
            sub_query_id="sq-001",
            url="https://doi.org/10.1038/test",
            title="Test",
            snippet="Test",
            backend="searxng",
            rank=1,
            score=0.5,
            authority=SourceAuthority.UNKNOWN,
            timestamp="2026-02-23T00:00:00Z",
            scholarly_metadata={"doi": "10.1038/test", "title": "Existing"},
        )
        result = await enrich_scholarly_result(sr)
        # Should return unchanged
        assert result["scholarly_metadata"]["title"] == "Existing"

    @pytest.mark.asyncio
    async def test_skip_if_no_doi(self):
        sr = SearchResult(
            id="sr-test",
            sub_query_id="sq-001",
            url="https://example.com/page",
            title="Test",
            snippet="Test",
            backend="searxng",
            rank=1,
            score=0.5,
            authority=SourceAuthority.UNKNOWN,
            timestamp="2026-02-23T00:00:00Z",
        )
        result = await enrich_scholarly_result(sr)
        assert "scholarly_metadata" not in result or result.get("scholarly_metadata") is None

    @pytest.mark.asyncio
    async def test_enriches_with_crossref_data(self):
        sr = SearchResult(
            id="sr-test",
            sub_query_id="sq-001",
            url="https://doi.org/10.1038/s41586-test",
            title="Test",
            snippet="Test",
            backend="searxng",
            rank=1,
            score=0.5,
            authority=SourceAuthority.UNKNOWN,
            timestamp="2026-02-23T00:00:00Z",
        )

        mock_work = {
            "DOI": "10.1038/s41586-test",
            "title": ["Enriched Title"],
            "author": [],
            "container-title": ["Nature"],
            "is-referenced-by-count": 50,
            "references-count": 10,
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": mock_work}
        mock_resp.raise_for_status = MagicMock()

        with patch("deep_research_swarm.backends.crossref.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await enrich_scholarly_result(sr)

        assert result["scholarly_metadata"]["doi"] == "10.1038/s41586-test"
        assert result["scholarly_metadata"]["venue"] == "Nature"

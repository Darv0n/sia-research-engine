"""Tests for backends/semantic_scholar.py — Semantic Scholar search (PR-03)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deep_research_swarm.backends.semantic_scholar import (
    SemanticScholarBackend,
    _build_provenance,
    _build_scholarly_metadata,
    _paper_url,
)
from deep_research_swarm.contracts import SearchBackend

# --- Sample S2 API response data ---

_SAMPLE_PAPER = {
    "paperId": "abc123def456",
    "title": "Advances in Quantum Entanglement",
    "abstract": "We present new results on quantum entanglement.",
    "year": 2024,
    "venue": "Physical Review Letters",
    "citationCount": 42,
    "referenceCount": 30,
    "isOpenAccess": True,
    "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    "url": "https://www.semanticscholar.org/paper/abc123def456",
    "externalIds": {
        "DOI": "10.1103/PhysRevLett.132.12345",
        "ArXiv": "2401.99999",
        "PubMed": "39876543",
    },
    "authors": [
        {"name": "Alice Smith"},
        {"name": "Bob Jones"},
    ],
}


def _make_search_response(papers: list[dict] | None = None) -> dict:
    return {"data": papers or [_SAMPLE_PAPER]}


class TestBuildScholarlyMetadata:
    def test_full_paper(self):
        sm = _build_scholarly_metadata(_SAMPLE_PAPER)
        assert sm["doi"] == "10.1103/PhysRevLett.132.12345"
        assert sm["arxiv_id"] == "2401.99999"
        assert sm["pmid"] == "39876543"
        assert sm["year"] == 2024
        assert sm["citation_count"] == 42
        assert sm["reference_count"] == 30
        assert sm["is_open_access"] is True
        assert sm["venue"] == "Physical Review Letters"
        assert len(sm["authors"]) == 2

    def test_minimal_paper(self):
        sm = _build_scholarly_metadata({"title": "Test"})
        assert sm["doi"] == ""
        assert sm["arxiv_id"] == ""
        assert sm["year"] == 0


class TestBuildProvenance:
    def test_basic(self):
        prov = _build_provenance(_SAMPLE_PAPER, "2026-02-23T00:00:00Z")
        assert prov["entity_id"] == "urn:s2:abc123def456"
        assert prov["source_kind"] == "scholarly"
        assert prov["extractor"] == "semantic_scholar"
        assert prov["content_hash"]  # Non-empty


class TestPaperUrl:
    def test_has_url(self):
        assert _paper_url(_SAMPLE_PAPER) == ("https://www.semanticscholar.org/paper/abc123def456")

    def test_fallback_to_doi(self):
        paper = {"externalIds": {"DOI": "10.1234/test"}}
        assert _paper_url(paper) == "https://doi.org/10.1234/test"

    def test_fallback_to_arxiv(self):
        paper = {"externalIds": {"ArXiv": "2401.12345"}}
        assert _paper_url(paper) == "https://arxiv.org/abs/2401.12345"

    def test_fallback_to_paper_id(self):
        paper = {"paperId": "xyz789"}
        assert "xyz789" in _paper_url(paper)


class TestSemanticScholarProtocol:
    def test_protocol_conformance(self):
        backend = SemanticScholarBackend()
        assert isinstance(backend, SearchBackend)
        assert backend.name == "semantic_scholar"


class TestSemanticScholarSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response()
        mock_resp.raise_for_status = MagicMock()

        backend = SemanticScholarBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            results = await backend.search("quantum entanglement", num_results=5)

        assert len(results) == 1
        assert results[0]["backend"] == "semantic_scholar"
        assert results[0]["scholarly_metadata"]["doi"] == "10.1103/PhysRevLett.132.12345"
        assert results[0]["provenance"]["entity_id"] == "urn:s2:abc123def456"

    @pytest.mark.asyncio
    async def test_search_empty_on_error(self):
        backend = SemanticScholarBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            results = await backend.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_populates_scholarly_metadata(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response()
        mock_resp.raise_for_status = MagicMock()

        backend = SemanticScholarBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            results = await backend.search("test")

        sm = results[0]["scholarly_metadata"]
        assert sm["authors"] == ["Alice Smith", "Bob Jones"]
        assert sm["open_access_url"] == "https://example.com/paper.pdf"


class TestGetPaperDetails:
    @pytest.mark.asyncio
    async def test_returns_paper_dict(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            **_SAMPLE_PAPER,
            "references": [{"paperId": "ref1", "title": "Ref 1"}],
            "citations": [{"paperId": "cit1", "title": "Cit 1"}],
        }
        mock_resp.raise_for_status = MagicMock()

        backend = SemanticScholarBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            details = await backend.get_paper_details("abc123def456")

        assert details is not None
        assert details["paperId"] == "abc123def456"
        assert len(details["references"]) == 1

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        backend = SemanticScholarBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            assert await backend.get_paper_details("nonexistent") is None


class TestSemanticScholarHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        backend = SemanticScholarBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        backend = SemanticScholarBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            assert await backend.health_check() is False

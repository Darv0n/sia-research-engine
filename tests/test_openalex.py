"""Tests for backends/openalex.py — OpenAlex scholarly search (PR-02)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deep_research_swarm.backends.openalex import (
    OpenAlexBackend,
    _build_scholarly_metadata,
    _extract_authors,
    _reconstruct_abstract,
)
from deep_research_swarm.contracts import SearchBackend

# --- Sample OpenAlex API response data ---

_SAMPLE_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1038/s41586-023-06768-0",
    "title": "Quantum entanglement in many-body systems",
    "publication_year": 2023,
    "cited_by_count": 150,
    "open_access": {
        "is_oa": True,
        "oa_url": "https://example.com/oa.pdf",
    },
    "primary_location": {
        "source": {"display_name": "Nature"},
    },
    "authorships": [
        {"author": {"display_name": "Alice Smith"}},
        {"author": {"display_name": "Bob Jones"}},
    ],
    "abstract_inverted_index": {
        "Quantum": [0],
        "entanglement": [1],
        "is": [2],
        "fundamental": [3],
    },
    "referenced_works": ["W123", "W456", "W789"],
    "ids": {
        "doi": "https://doi.org/10.1038/s41586-023-06768-0",
        "pmid": "12345678",
    },
}


def _make_search_response(works: list[dict] | None = None) -> dict:
    return {"results": works or [_SAMPLE_WORK]}


class TestReconstructAbstract:
    def test_basic(self):
        inverted = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(inverted) == "Hello world"

    def test_out_of_order(self):
        inverted = {"world": [1], "Hello": [0]}
        assert _reconstruct_abstract(inverted) == "Hello world"

    def test_repeated_word(self):
        inverted = {"the": [0, 2], "cat": [1], "dog": [3]}
        assert _reconstruct_abstract(inverted) == "the cat the dog"

    def test_empty(self):
        assert _reconstruct_abstract(None) == ""
        assert _reconstruct_abstract({}) == ""


class TestExtractAuthors:
    def test_basic(self):
        authorships = [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ]
        assert _extract_authors(authorships) == ["Alice", "Bob"]

    def test_empty(self):
        assert _extract_authors([]) == []

    def test_missing_name(self):
        authorships = [{"author": {"display_name": ""}}, {"author": {"display_name": "Bob"}}]
        assert _extract_authors(authorships) == ["Bob"]


class TestBuildScholarlyMetadata:
    def test_full_work(self):
        sm = _build_scholarly_metadata(_SAMPLE_WORK)
        assert sm["doi"] == "10.1038/s41586-023-06768-0"
        assert sm["year"] == 2023
        assert sm["citation_count"] == 150
        assert sm["reference_count"] == 3
        assert sm["is_open_access"] is True
        assert sm["venue"] == "Nature"
        assert len(sm["authors"]) == 2
        assert "Quantum" in sm["abstract"]

    def test_minimal_work(self):
        sm = _build_scholarly_metadata({"title": "Test"})
        assert sm["doi"] == ""
        assert sm["year"] == 0
        assert sm["citation_count"] == 0


class TestOpenAlexProtocol:
    def test_protocol_conformance(self):
        backend = OpenAlexBackend(email="test@example.com")
        assert isinstance(backend, SearchBackend)
        assert backend.name == "openalex"


class TestOpenAlexSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response()
        mock_resp.raise_for_status = MagicMock()

        backend = OpenAlexBackend(email="test@example.com")
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            results = await backend.search("quantum entanglement", num_results=5)

        assert len(results) == 1
        assert results[0]["backend"] == "openalex"
        assert results[0]["scholarly_metadata"]["doi"] == "10.1038/s41586-023-06768-0"
        assert results[0]["provenance"]["source_kind"] == "scholarly"

    @pytest.mark.asyncio
    async def test_search_empty_on_error(self):
        backend = OpenAlexBackend(email="test@example.com")
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            results = await backend.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_populates_provenance(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response()
        mock_resp.raise_for_status = MagicMock()

        backend = OpenAlexBackend(email="test@example.com")
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            results = await backend.search("test")

        prov = results[0]["provenance"]
        assert prov["extractor"] == "openalex"
        assert prov["license_tag"] == "CC0"
        assert prov["entity_id"].startswith("urn:openalex:")


class TestOpenAlexHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        backend = OpenAlexBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        backend = OpenAlexBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            assert await backend.health_check() is False

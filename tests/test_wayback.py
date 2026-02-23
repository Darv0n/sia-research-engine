"""Tests for backends/wayback.py — Wayback Machine backend (PR-05)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deep_research_swarm.backends.wayback import WaybackBackend
from deep_research_swarm.contracts import ArchiveCapture, SearchBackend

# Sample CDX API response (header + data rows)
_SAMPLE_CDX = [
    ["timestamp", "original", "statuscode", "mimetype"],
    ["20240101120000", "https://example.com/page", "200", "text/html"],
    ["20231215080000", "https://example.com/page", "200", "text/html"],
]


class TestWaybackProtocol:
    def test_protocol_conformance(self):
        backend = WaybackBackend()
        assert isinstance(backend, SearchBackend)
        assert backend.name == "wayback"


class TestWaybackSearch:
    @pytest.mark.asyncio
    async def test_non_url_query_returns_empty(self):
        """D6: search() returns empty for non-URL queries."""
        backend = WaybackBackend()
        results = await backend.search("quantum entanglement")
        assert results == []

    @pytest.mark.asyncio
    async def test_url_query_returns_results(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_CDX

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            results = await backend.search("https://example.com/page", num_results=5)

        assert len(results) == 2
        assert results[0]["backend"] == "wayback"
        assert "web.archive.org" in results[0]["url"]
        assert results[0]["provenance"]["source_kind"] == "archive"

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self):
        backend = WaybackBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            results = await backend.search("https://example.com/page")
        assert results == []


class TestGetCaptures:
    @pytest.mark.asyncio
    async def test_returns_captures(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_CDX

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            captures = await backend.get_captures("https://example.com/page")

        assert len(captures) == 2
        assert captures[0]["capture_timestamp"] == "20240101120000"
        assert captures[0]["original_url"] == "https://example.com/page"
        assert "id_" in captures[0]["archive_url"]

    @pytest.mark.asyncio
    async def test_empty_cdx_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [["timestamp", "original", "statuscode", "mimetype"]]

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            captures = await backend.get_captures("https://nonexistent.example.com")
        assert captures == []

    @pytest.mark.asyncio
    async def test_error_returns_empty(self):
        backend = WaybackBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            captures = await backend.get_captures("https://example.com")
        assert captures == []


class TestLookupUrl:
    @pytest.mark.asyncio
    async def test_returns_most_recent(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_CDX

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            capture = await backend.lookup_url("https://example.com/page")

        assert capture is not None
        assert capture["capture_timestamp"] == "20240101120000"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_archived(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [["timestamp", "original", "statuscode", "mimetype"]]

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            capture = await backend.lookup_url("https://nonexistent.example.com")
        assert capture is None


class TestFetchArchivedContent:
    @pytest.mark.asyncio
    async def test_fetches_content(self):
        capture = ArchiveCapture(
            original_url="https://example.com/page",
            archive_url="https://web.archive.org/web/20240101120000id_/https://example.com/page",
            capture_timestamp="20240101120000",
            status_code=200,
            content_type="text/html",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>Archived content</body></html>"

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            content = await backend.fetch_archived_content(capture)
        assert "Archived content" in content

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        capture = ArchiveCapture(
            original_url="https://example.com",
            archive_url="https://web.archive.org/web/2024id_/https://example.com",
            capture_timestamp="2024",
            status_code=200,
            content_type="text/html",
        )
        backend = WaybackBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            content = await backend.fetch_archived_content(capture)
        assert content == ""


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        backend = WaybackBackend()
        with patch.object(backend._client, "get", AsyncMock(return_value=mock_resp)):
            assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        backend = WaybackBackend()
        with patch.object(
            backend._client, "get", AsyncMock(side_effect=httpx.ConnectError("fail"))
        ):
            assert await backend.health_check() is False

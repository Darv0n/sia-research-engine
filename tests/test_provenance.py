"""Tests for provenance tracking (PR-06)."""

from __future__ import annotations

from deep_research_swarm.contracts import SearchResult, SourceAuthority
from deep_research_swarm.reporting.provenance import render_provenance_section
from deep_research_swarm.scoring.provenance import (
    attach_provenance_to_result,
    build_provenance,
)


class TestBuildProvenance:
    def test_basic(self):
        prov = build_provenance(
            url="https://example.com/page",
            extractor="trafilatura",
            content="Some extracted content here.",
        )
        assert prov["entity_id"] == "urn:url:https://example.com/page"
        assert prov["source_kind"] == "web"
        assert prov["extractor"] == "trafilatura"
        assert prov["content_hash"]  # Non-empty
        assert prov["capture_timestamp"] == ""

    def test_archive_source_kind(self):
        prov = build_provenance(
            url="https://example.com",
            extractor="wayback",
            content="Archived text",
            source_kind="archive",
            capture_timestamp="20240101120000",
        )
        assert prov["source_kind"] == "archive"
        assert prov["capture_timestamp"] == "20240101120000"

    def test_empty_content_empty_hash(self):
        prov = build_provenance(
            url="https://example.com",
            extractor="none",
            content="",
        )
        assert prov["content_hash"] == ""

    def test_content_hash_deterministic(self):
        prov1 = build_provenance(url="a", extractor="b", content="test")
        prov2 = build_provenance(url="a", extractor="b", content="test")
        assert prov1["content_hash"] == prov2["content_hash"]


class TestAttachProvenanceToResult:
    def _make_sr(self, **kwargs) -> SearchResult:
        base = dict(
            id="sr-test",
            sub_query_id="sq-001",
            url="https://example.com/page",
            title="Test",
            snippet="Test snippet",
            backend="searxng",
            rank=1,
            score=0.5,
            authority=SourceAuthority.UNKNOWN,
            timestamp="2026-02-23T00:00:00Z",
        )
        base.update(kwargs)
        return SearchResult(**base)

    def test_creates_provenance_when_absent(self):
        sr = self._make_sr()
        result = attach_provenance_to_result(sr, extractor="trafilatura", content="text")
        assert "provenance" in result
        assert result["provenance"]["extractor"] == "trafilatura"
        assert result["provenance"]["source_kind"] == "web"

    def test_updates_hash_when_existing(self):
        sr = self._make_sr(
            provenance={
                "entity_id": "urn:s2:abc",
                "source_url": "https://example.com",
                "source_kind": "scholarly",
                "fetched_at": "2026-02-23T00:00:00Z",
                "extractor": "semantic_scholar",
                "license_tag": "unknown",
                "capture_timestamp": "",
                "content_hash": "old_hash",
            }
        )
        result = attach_provenance_to_result(sr, extractor="trafilatura", content="new content")
        # Should keep existing provenance but update hash
        assert result["provenance"]["extractor"] == "semantic_scholar"
        assert result["provenance"]["content_hash"] != "old_hash"

    def test_wayback_source_kind(self):
        sr = self._make_sr()
        result = attach_provenance_to_result(sr, extractor="wayback+trafilatura", content="text")
        assert result["provenance"]["source_kind"] == "archive"

    def test_pdf_source_kind(self):
        sr = self._make_sr()
        result = attach_provenance_to_result(sr, extractor="pdf", content="text")
        assert result["provenance"]["source_kind"] == "pdf"


class TestRenderProvenanceSection:
    def test_renders_table(self):
        results = [
            SearchResult(
                id="sr-1",
                sub_query_id="sq-001",
                url="https://example.com/page",
                title="Example Page",
                snippet="",
                backend="searxng",
                rank=1,
                score=0.5,
                authority=SourceAuthority.UNKNOWN,
                timestamp="2026-02-23T00:00:00Z",
                provenance={
                    "entity_id": "urn:url:https://example.com/page",
                    "source_url": "https://example.com/page",
                    "source_kind": "web",
                    "fetched_at": "2026-02-23T00:00:00Z",
                    "extractor": "trafilatura",
                    "license_tag": "unknown",
                    "capture_timestamp": "",
                    "content_hash": "abcdef1234567890",
                },
            )
        ]
        table = render_provenance_section(results)
        assert "| Source | Kind | Extractor | Hash |" in table
        assert "trafilatura" in table
        assert "abcdef12" in table  # Truncated hash

    def test_empty_without_provenance(self):
        results = [
            SearchResult(
                id="sr-1",
                sub_query_id="sq-001",
                url="https://example.com",
                title="Test",
                snippet="",
                backend="searxng",
                rank=1,
                score=0.5,
                authority=SourceAuthority.UNKNOWN,
                timestamp="2026-02-23T00:00:00Z",
            )
        ]
        assert render_provenance_section(results) == ""

    def test_deduplicates_by_url(self):
        prov = {
            "entity_id": "urn:url:x",
            "source_url": "https://example.com",
            "source_kind": "web",
            "fetched_at": "2026-02-23T00:00:00Z",
            "extractor": "crawl4ai",
            "license_tag": "unknown",
            "capture_timestamp": "",
            "content_hash": "abc",
        }
        results = [
            SearchResult(
                id="sr-1",
                sub_query_id="sq-001",
                url="https://example.com",
                title="Test",
                snippet="",
                backend="searxng",
                rank=1,
                score=0.5,
                authority=SourceAuthority.UNKNOWN,
                timestamp="2026-02-23T00:00:00Z",
                provenance=prov,
            ),
            SearchResult(
                id="sr-2",
                sub_query_id="sq-002",
                url="https://example.com",
                title="Test Dupe",
                snippet="",
                backend="exa",
                rank=1,
                score=0.5,
                authority=SourceAuthority.UNKNOWN,
                timestamp="2026-02-23T00:00:00Z",
                provenance=prov,
            ),
        ]
        table = render_provenance_section(results)
        # Should only appear once
        assert table.count("example.com") == 1

    def test_empty_list(self):
        assert render_provenance_section([]) == ""

"""Tests for memory/incremental.py — content-hash-based diffing (PR-10, OE2)."""

from __future__ import annotations

from deep_research_swarm.contracts import ExtractedContent, ProvenanceRecord
from deep_research_swarm.memory.incremental import (
    compute_content_diff,
    compute_diff_summary,
    filter_unchanged_sources,
)


def _make_prov(url: str, content_hash: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        entity_id=f"urn:url:{url}",
        source_url=url,
        source_kind="web",
        fetched_at="2026-02-23T00:00:00Z",
        extractor="crawl4ai",
        license_tag="unknown",
        capture_timestamp="",
        content_hash=content_hash,
    )


def _make_ec(url: str) -> ExtractedContent:
    return ExtractedContent(
        id=f"ec-{url[-4:]}",
        search_result_id="sr-001",
        url=url,
        title="Test",
        content="content",
        content_length=7,
        extractor_used="crawl4ai",
        extraction_success=True,
        error=None,
    )


# --- compute_content_diff ---


class TestComputeContentDiff:
    def test_new_source(self):
        current = [_make_prov("https://example.com/new", "abc123")]
        previous: list[ProvenanceRecord] = []
        diff = compute_content_diff(current, previous)
        assert diff["https://example.com/new"] == "new"

    def test_unchanged_source(self):
        current = [_make_prov("https://example.com/page", "abc123")]
        previous = [_make_prov("https://example.com/page", "abc123")]
        diff = compute_content_diff(current, previous)
        assert diff["https://example.com/page"] == "unchanged"

    def test_changed_source(self):
        current = [_make_prov("https://example.com/page", "new_hash")]
        previous = [_make_prov("https://example.com/page", "old_hash")]
        diff = compute_content_diff(current, previous)
        assert diff["https://example.com/page"] == "changed"

    def test_empty_hash_treated_as_new(self):
        current = [_make_prov("https://example.com/page", "")]
        previous = [_make_prov("https://example.com/page", "abc123")]
        diff = compute_content_diff(current, previous)
        assert diff["https://example.com/page"] == "new"

    def test_mixed_statuses(self):
        current = [
            _make_prov("https://a.com", "hash_a"),
            _make_prov("https://b.com", "new_hash_b"),
            _make_prov("https://c.com", "hash_c"),
        ]
        previous = [
            _make_prov("https://a.com", "hash_a"),
            _make_prov("https://b.com", "old_hash_b"),
        ]
        diff = compute_content_diff(current, previous)
        assert diff["https://a.com"] == "unchanged"
        assert diff["https://b.com"] == "changed"
        assert diff["https://c.com"] == "new"

    def test_empty_inputs(self):
        assert compute_content_diff([], []) == {}

    def test_only_previous(self):
        """Current empty, previous has sources — returns empty (nothing current)."""
        diff = compute_content_diff([], [_make_prov("https://a.com", "hash")])
        assert diff == {}


# --- filter_unchanged_sources ---


class TestFilterUnchangedSources:
    def test_unchanged_skipped(self):
        ecs = [_make_ec("https://a.com"), _make_ec("https://b.com")]
        diff = {"https://a.com": "unchanged", "https://b.com": "new"}
        to_process, skipped = filter_unchanged_sources(ecs, diff)
        assert len(to_process) == 1
        assert to_process[0]["url"] == "https://b.com"
        assert len(skipped) == 1
        assert skipped[0]["url"] == "https://a.com"

    def test_changed_processed(self):
        ecs = [_make_ec("https://a.com")]
        diff = {"https://a.com": "changed"}
        to_process, skipped = filter_unchanged_sources(ecs, diff)
        assert len(to_process) == 1
        assert len(skipped) == 0

    def test_unknown_url_treated_as_new(self):
        ecs = [_make_ec("https://unknown.com")]
        diff = {}  # URL not in diff
        to_process, skipped = filter_unchanged_sources(ecs, diff)
        assert len(to_process) == 1
        assert len(skipped) == 0

    def test_empty_inputs(self):
        to_process, skipped = filter_unchanged_sources([], {})
        assert to_process == []
        assert skipped == []

    def test_all_unchanged(self):
        ecs = [_make_ec("https://a.com"), _make_ec("https://b.com")]
        diff = {"https://a.com": "unchanged", "https://b.com": "unchanged"}
        to_process, skipped = filter_unchanged_sources(ecs, diff)
        assert len(to_process) == 0
        assert len(skipped) == 2


# --- compute_diff_summary ---


class TestComputeDiffSummary:
    def test_basic_summary(self):
        diff = {
            "https://a.com": "new",
            "https://b.com": "unchanged",
            "https://c.com": "changed",
            "https://d.com": "new",
        }
        summary = compute_diff_summary(diff)
        assert summary["new"] == 2
        assert summary["unchanged"] == 1
        assert summary["changed"] == 1

    def test_empty_diff(self):
        summary = compute_diff_summary({})
        assert summary == {"new": 0, "changed": 0, "unchanged": 0}

    def test_all_same_status(self):
        diff = {"https://a.com": "new", "https://b.com": "new"}
        summary = compute_diff_summary(diff)
        assert summary["new"] == 2
        assert summary["changed"] == 0
        assert summary["unchanged"] == 0

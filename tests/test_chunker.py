"""Tests for extractors/chunker.py — passage chunking (PR-10)."""

from __future__ import annotations

import hashlib

from deep_research_swarm.contracts import ExtractedContent, ScoredDocument, SourceAuthority
from deep_research_swarm.extractors.chunker import (
    _estimate_tokens,
    _make_passage_id,
    chunk_all_documents,
    chunk_document,
)


def _make_ec(content: str, url: str = "https://example.com") -> ExtractedContent:
    return ExtractedContent(
        id="ec-test",
        search_result_id="sr-test",
        url=url,
        title="Test",
        content=content,
        content_length=len(content),
        extractor_used="trafilatura",
        extraction_success=True,
        error=None,
    )


def _make_sd(url: str = "https://example.com") -> ScoredDocument:
    return ScoredDocument(
        id="sd-test",
        url=url,
        title="Test",
        content="",
        rrf_score=0.5,
        authority=SourceAuthority.UNKNOWN,
        authority_score=0.5,
        combined_score=0.5,
        sub_query_ids=["sq-001"],
    )


class TestMakePassageId:
    def test_deterministic(self):
        """Same input produces same ID (D1, I1)."""
        id1 = _make_passage_id("source-a", 0)
        id2 = _make_passage_id("source-a", 0)
        assert id1 == id2

    def test_prefix(self):
        """IDs start with sp- prefix."""
        pid = _make_passage_id("source-a", 0)
        assert pid.startswith("sp-")
        assert len(pid) == 11  # "sp-" + 8 hex chars

    def test_different_position_different_id(self):
        id1 = _make_passage_id("source-a", 0)
        id2 = _make_passage_id("source-a", 1)
        assert id1 != id2

    def test_different_source_different_id(self):
        id1 = _make_passage_id("source-a", 0)
        id2 = _make_passage_id("source-b", 0)
        assert id1 != id2

    def test_matches_manual_hash(self):
        source_id = "https://example.comhash123"
        position = 0
        raw = (source_id + str(position)).encode()
        expected = f"sp-{hashlib.sha256(raw).hexdigest()[:8]}"
        assert _make_passage_id(source_id, position) == expected


class TestEstimateTokens:
    def test_single_word(self):
        assert _estimate_tokens("hello") == 1

    def test_ten_words(self):
        text = " ".join(["word"] * 10)
        assert _estimate_tokens(text) == 13  # 10 * 1.3

    def test_empty_returns_one(self):
        assert _estimate_tokens("") == 1


class TestChunkDocument:
    def test_empty_content_returns_empty(self):
        ec = _make_ec("")
        sd = _make_sd()
        assert chunk_document(ec, sd) == []

    def test_short_document_single_passage(self):
        """Documents shorter than target_tokens return one passage."""
        ec = _make_ec("Short document about quantum physics.")
        sd = _make_sd()
        passages = chunk_document(ec, sd, target_tokens=300)
        assert len(passages) == 1
        assert passages[0]["position"] == 0
        assert passages[0]["claim_ids"] == []

    def test_long_document_multiple_passages(self):
        """Long document produces multiple passages."""
        # Create content with multiple paragraphs
        paras = [("Paragraph about topic number %d. " % i) * 10 for i in range(10)]
        content = "\n\n".join(paras)
        ec = _make_ec(content)
        sd = _make_sd()
        passages = chunk_document(ec, sd, target_tokens=50, overlap_tokens=0)
        assert len(passages) > 1

    def test_heading_splitting(self):
        """Markdown headings create section boundaries."""
        content = (
            "## Introduction\n\nFirst section content here.\n\n"
            "## Methods\n\nSecond section content here.\n\n"
            "## Results\n\nThird section content here."
        )
        ec = _make_ec(content)
        sd = _make_sd()
        passages = chunk_document(ec, sd, target_tokens=300)
        headings = [p["heading_context"] for p in passages]
        assert "## Introduction" in headings
        assert "## Methods" in headings

    def test_code_block_not_split(self):
        """Code blocks are protected from splitting."""
        content = (
            "Some text before.\n\n"
            "```python\ndef hello():\n    return 'world'\n```\n\n"
            "Some text after."
        )
        ec = _make_ec(content)
        sd = _make_sd()
        passages = chunk_document(ec, sd, target_tokens=300)
        # Code block should be intact in one passage
        code_passages = [p for p in passages if "def hello" in p["content"]]
        assert len(code_passages) >= 1
        assert "return 'world'" in code_passages[0]["content"]

    def test_table_not_split(self):
        """Markdown tables are protected from splitting."""
        content = (
            "Some text before.\n\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| val 1 | val 2 |\n"
            "| val 3 | val 4 |\n\n"
            "Some text after."
        )
        ec = _make_ec(content)
        sd = _make_sd()
        passages = chunk_document(ec, sd, target_tokens=300)
        table_passages = [p for p in passages if "Col A" in p["content"]]
        assert len(table_passages) >= 1
        assert "val 4" in table_passages[0]["content"]

    def test_passage_ids_are_deterministic(self):
        """Same content produces same passage IDs across calls (D1)."""
        ec = _make_ec("Test content for determinism check.")
        sd = _make_sd()
        p1 = chunk_document(ec, sd)
        p2 = chunk_document(ec, sd)
        assert [p["id"] for p in p1] == [p["id"] for p in p2]

    def test_source_id_includes_content_hash(self):
        """source_id = url + content_hash (content-stable)."""
        ec = _make_ec("Some unique content here.")
        sd = _make_sd()
        passages = chunk_document(ec, sd)
        assert len(passages) >= 1
        # source_id should start with URL
        assert passages[0]["source_id"].startswith("https://example.com")
        # And be longer than just the URL (has hash appended)
        assert len(passages[0]["source_id"]) > len("https://example.com")


class TestChunkAllDocuments:
    def test_matches_by_url(self):
        """Only chunks documents where URL matches a scored document."""
        ec1 = _make_ec("Content one.", url="https://a.com")
        ec2 = _make_ec("Content two.", url="https://b.com")
        sd1 = _make_sd(url="https://a.com")
        # ec2 has no matching scored doc
        passages = chunk_all_documents([ec1, ec2], [sd1])
        urls = {p["source_url"] for p in passages}
        assert "https://a.com" in urls
        assert "https://b.com" not in urls

    def test_skips_failed_extractions(self):
        """Documents with extraction_success=False are skipped."""
        ec = ExtractedContent(
            id="ec-fail",
            search_result_id="sr-fail",
            url="https://example.com",
            title="Failed",
            content="Some content",
            content_length=12,
            extractor_used="trafilatura",
            extraction_success=False,
            error="timeout",
        )
        sd = _make_sd()
        assert chunk_all_documents([ec], [sd]) == []

    def test_empty_inputs(self):
        assert chunk_all_documents([], []) == []

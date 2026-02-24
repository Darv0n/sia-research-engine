"""Tests for scoring/claim_graph.py — claim extraction and passage linking (PR-08)."""

from __future__ import annotations

from deep_research_swarm.contracts import (
    Confidence,
    GraderScores,
    SectionDraft,
    SourcePassage,
)
from deep_research_swarm.scoring.claim_graph import (
    _claim_id,
    extract_claims_from_section,
    link_claims_to_passages,
    populate_claim_ids,
)


def _make_passage(
    content: str,
    *,
    pid: str = "sp-test0001",
    source_id: str = "src-a",
    position: int = 0,
) -> SourcePassage:
    return SourcePassage(
        id=pid,
        source_id=source_id,
        source_url="https://example.com",
        content=content,
        position=position,
        char_offset=0,
        token_count=len(content.split()),
        heading_context="",
        claim_ids=[],
    )


def _make_section(
    content: str,
    *,
    section_id: str = "sec-001",
    heading: str = "Test Section",
    citation_ids: list[str] | None = None,
) -> SectionDraft:
    return SectionDraft(
        id=section_id,
        heading=heading,
        content=content,
        citation_ids=citation_ids or [],
        confidence_score=0.8,
        confidence_level=Confidence.HIGH,
        grader_scores=GraderScores(relevance=0.8, hallucination=0.8, quality=0.8),
    )


# --- _claim_id ---


class TestClaimId:
    def test_deterministic(self):
        """Same inputs produce same ID."""
        assert _claim_id("sec-001", 0) == _claim_id("sec-001", 0)

    def test_different_for_different_indices(self):
        """Different claim indices produce different IDs."""
        assert _claim_id("sec-001", 0) != _claim_id("sec-001", 1)

    def test_different_for_different_sections(self):
        """Different section IDs produce different claim IDs."""
        assert _claim_id("sec-001", 0) != _claim_id("sec-002", 0)

    def test_format(self):
        """Claim ID starts with 'cl-' prefix."""
        cid = _claim_id("sec-001", 0)
        assert cid.startswith("cl-")
        assert len(cid) == 11  # "cl-" + 8 hex chars


# --- extract_claims_from_section ---


class TestExtractClaims:
    def test_single_claim(self):
        section = _make_section(
            "Quantum entanglement enables communication [1].",
            citation_ids=["[1]"],
        )
        claims = extract_claims_from_section(section)
        assert len(claims) == 1
        assert claims[0]["citation_ids"] == ["[1]"]
        assert "Quantum entanglement" in claims[0]["text"]
        assert "[1]" not in claims[0]["text"]  # citation marker removed

    def test_multiple_claims(self):
        section = _make_section(
            "Entanglement is fundamental [1]. Bell tests confirm this [2].",
            citation_ids=["[1]", "[2]"],
        )
        claims = extract_claims_from_section(section)
        assert len(claims) == 2

    def test_multiple_citations_in_one_sentence(self):
        section = _make_section(
            "Research shows results [1] [2].",
            citation_ids=["[1]", "[2]"],
        )
        claims = extract_claims_from_section(section)
        assert len(claims) == 1
        assert "[1]" in claims[0]["citation_ids"]
        assert "[2]" in claims[0]["citation_ids"]

    def test_no_citations(self):
        section = _make_section("Plain text without citations.")
        claims = extract_claims_from_section(section)
        assert claims == []

    def test_empty_content(self):
        section = _make_section("")
        claims = extract_claims_from_section(section)
        assert claims == []

    def test_claim_has_section_id(self):
        section = _make_section("Claim text [1].", section_id="sec-xyz")
        claims = extract_claims_from_section(section)
        assert claims[0]["section_id"] == "sec-xyz"

    def test_claim_has_section_heading(self):
        section = _make_section("Claim text [1].", heading="Introduction")
        claims = extract_claims_from_section(section)
        assert claims[0]["section_heading"] == "Introduction"

    def test_claim_ids_are_unique(self):
        section = _make_section("First claim [1]. Second claim [2]. Third claim [3].")
        claims = extract_claims_from_section(section)
        ids = [c["id"] for c in claims]
        assert len(ids) == len(set(ids))

    def test_deterministic(self):
        section = _make_section("Claim text [1].")
        c1 = extract_claims_from_section(section)
        c2 = extract_claims_from_section(section)
        assert c1 == c2


# --- link_claims_to_passages ---


class TestLinkClaimsToPassages:
    def test_basic_linking(self):
        claims = [
            {"id": "cl-001", "citation_ids": ["[1]"], "text": "claim", "section_id": "s1",
             "section_heading": "h1"},
        ]
        c2p = {"[1]": ["sp-001", "sp-002"]}
        result = link_claims_to_passages(claims, c2p)
        assert result["cl-001"] == ["sp-001", "sp-002"]

    def test_multiple_citations(self):
        claims = [
            {"id": "cl-001", "citation_ids": ["[1]", "[2]"], "text": "claim",
             "section_id": "s1", "section_heading": "h1"},
        ]
        c2p = {"[1]": ["sp-001"], "[2]": ["sp-002"]}
        result = link_claims_to_passages(claims, c2p)
        assert "sp-001" in result["cl-001"]
        assert "sp-002" in result["cl-001"]

    def test_deduplicates_passage_ids(self):
        claims = [
            {"id": "cl-001", "citation_ids": ["[1]", "[2]"], "text": "claim",
             "section_id": "s1", "section_heading": "h1"},
        ]
        # Both citations point to same passage
        c2p = {"[1]": ["sp-001"], "[2]": ["sp-001"]}
        result = link_claims_to_passages(claims, c2p)
        assert result["cl-001"] == ["sp-001"]  # deduplicated

    def test_missing_citation(self):
        claims = [
            {"id": "cl-001", "citation_ids": ["[99]"], "text": "claim",
             "section_id": "s1", "section_heading": "h1"},
        ]
        c2p = {"[1]": ["sp-001"]}
        result = link_claims_to_passages(claims, c2p)
        assert result["cl-001"] == []

    def test_empty_claims(self):
        result = link_claims_to_passages([], {"[1]": ["sp-001"]})
        assert result == {}


# --- populate_claim_ids ---


class TestPopulateClaimIds:
    def test_populates_claim_ids(self):
        passages = [
            _make_passage("content a", pid="sp-001"),
            _make_passage("content b", pid="sp-002"),
        ]
        claims = [
            {"id": "cl-001", "citation_ids": ["[1]"], "text": "claim 1",
             "section_id": "s1", "section_heading": "h1"},
            {"id": "cl-002", "citation_ids": ["[2]"], "text": "claim 2",
             "section_id": "s1", "section_heading": "h1"},
        ]
        c2p = {"[1]": ["sp-001"], "[2]": ["sp-001", "sp-002"]}
        updated = populate_claim_ids(passages, claims, c2p)
        # sp-001 is cited by both claims
        p1 = next(p for p in updated if p["id"] == "sp-001")
        assert "cl-001" in p1["claim_ids"]
        assert "cl-002" in p1["claim_ids"]
        # sp-002 is cited by cl-002 only
        p2 = next(p for p in updated if p["id"] == "sp-002")
        assert p2["claim_ids"] == ["cl-002"]

    def test_does_not_mutate_input(self):
        passages = [_make_passage("content", pid="sp-001")]
        claims = [
            {"id": "cl-001", "citation_ids": ["[1]"], "text": "claim",
             "section_id": "s1", "section_heading": "h1"},
        ]
        c2p = {"[1]": ["sp-001"]}
        _ = populate_claim_ids(passages, claims, c2p)
        assert passages[0]["claim_ids"] == []  # original unchanged

    def test_empty_passages(self):
        result = populate_claim_ids([], [], {})
        assert result == []

    def test_no_claims_leaves_empty(self):
        passages = [_make_passage("content", pid="sp-001")]
        updated = populate_claim_ids(passages, [], {})
        assert updated[0]["claim_ids"] == []

    def test_preserves_passage_fields(self):
        passages = [_make_passage("test content", pid="sp-001", source_id="src-x")]
        updated = populate_claim_ids(passages, [], {})
        assert updated[0]["source_id"] == "src-x"
        assert updated[0]["content"] == "test content"
        assert updated[0]["source_url"] == "https://example.com"

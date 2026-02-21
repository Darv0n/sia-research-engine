"""Tests for reporting.evidence_map — claim extraction and table rendering."""

from __future__ import annotations

from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    GraderScores,
    SectionDraft,
    SourceAuthority,
)
from deep_research_swarm.reporting.evidence_map import extract_claims, render_evidence_map


def _make_section(
    heading: str = "Test Section",
    content: str = "This is a test claim [1].",
    citation_ids: list[str] | None = None,
    confidence_score: float = 0.85,
) -> SectionDraft:
    return SectionDraft(
        id="sec-001",
        heading=heading,
        content=content,
        citation_ids=citation_ids or ["[1]"],
        confidence_score=confidence_score,
        confidence_level=Confidence.HIGH,
        grader_scores=GraderScores(relevance=0.9, hallucination=0.85, quality=0.8),
    )


def _make_citation(
    cid: str = "[1]",
    title: str = "Source One",
    url: str = "https://example.com/1",
    authority: SourceAuthority = SourceAuthority.INSTITUTIONAL,
) -> Citation:
    return Citation(
        id=cid,
        url=url,
        title=title,
        authority=authority,
        accessed="2026-02-21T00:00:00Z",
        used_in_sections=["Test Section"],
    )


class TestExtractClaims:
    def test_extracts_cited_sentence(self):
        text = "Quantum entanglement is well established [1]. More text here."
        sections = [_make_section(content=text)]
        citations = [_make_citation()]
        claims = extract_claims(sections, citations)
        assert len(claims) == 1
        assert "[1]" in claims[0]["claim"]
        assert claims[0]["source_title"] == "Source One"
        assert claims[0]["section"] == "Test Section"

    def test_uncited_sentences_ignored(self):
        sections = [_make_section(content="No citations in this sentence at all.")]
        citations = [_make_citation()]
        claims = extract_claims(sections, citations)
        assert len(claims) == 0

    def test_max_claims_limit(self):
        content = " ".join(f"Claim number {i} is documented [1]." for i in range(30))
        sections = [_make_section(content=content)]
        citations = [_make_citation()]
        claims = extract_claims(sections, citations, max_claims=5)
        assert len(claims) == 5

    def test_section_heading_included(self):
        sections = [_make_section(heading="Methodology")]
        citations = [_make_citation()]
        claims = extract_claims(sections, citations)
        assert claims[0]["section"] == "Methodology"

    def test_confidence_included(self):
        sections = [_make_section(confidence_score=0.72)]
        citations = [_make_citation()]
        claims = extract_claims(sections, citations)
        assert claims[0]["confidence"] == 0.72

    def test_multiple_citations_in_sentence(self):
        sections = [_make_section(content="Both sources agree [1] and confirm [2] the hypothesis.")]
        citations = [_make_citation("[1]"), _make_citation("[2]", title="Source Two")]
        claims = extract_claims(sections, citations)
        assert len(claims) == 2
        titles = {c["source_title"] for c in claims}
        assert titles == {"Source One", "Source Two"}

    def test_multiple_sections(self):
        text_a = "The first substantial claim is documented in the literature [1]."
        text_b = "The second substantial claim is also well supported [2]."
        sections = [
            _make_section(heading="Section A", content=text_a),
            _make_section(heading="Section B", content=text_b),
        ]
        citations = [_make_citation("[1]"), _make_citation("[2]", title="Source Two")]
        claims = extract_claims(sections, citations)
        assert len(claims) == 2
        assert claims[0]["section"] == "Section A"
        assert claims[1]["section"] == "Section B"

    def test_empty_sections(self):
        claims = extract_claims([], [])
        assert claims == []

    def test_citation_not_found_skipped(self):
        sections = [_make_section(content="This references missing [99].")]
        citations = [_make_citation("[1]")]
        claims = extract_claims(sections, citations)
        assert len(claims) == 0


class TestRenderEvidenceMap:
    def test_renders_table_header(self):
        claims = [
            {
                "claim": "Test claim [1].",
                "section": "Intro",
                "citation_id": "[1]",
                "source_title": "Source",
                "source_url": "https://example.com",
                "authority": SourceAuthority.INSTITUTIONAL,
                "confidence": 0.85,
            }
        ]
        table = render_evidence_map(claims)
        assert "| Claim |" in table
        assert "| Source |" in table
        assert "| Authority |" in table
        assert "| Confidence |" in table
        assert "0.85" in table

    def test_pipe_escaping(self):
        claims = [
            {
                "claim": "Value A | Value B [1].",
                "section": "Test",
                "citation_id": "[1]",
                "source_title": "Source | Title",
                "source_url": "https://example.com",
                "authority": "institutional",
                "confidence": 0.80,
            }
        ]
        table = render_evidence_map(claims)
        # Pipes in content should be escaped
        assert "\\|" in table

    def test_empty_claims_returns_empty_string(self):
        assert render_evidence_map([]) == ""

    def test_authority_enum_value_rendered(self):
        claims = [
            {
                "claim": "Test [1].",
                "section": "Test",
                "citation_id": "[1]",
                "source_title": "Source",
                "source_url": "https://example.com",
                "authority": SourceAuthority.PROFESSIONAL,
                "confidence": 0.90,
            }
        ]
        table = render_evidence_map(claims)
        assert "professional" in table

"""Tests for report rendering."""

import yaml

from deep_research_swarm.contracts import (
    Citation,
    IterationRecord,
    ResearchGap,
    SourceAuthority,
)
from deep_research_swarm.reporting.citations import (
    build_bibliography,
    deduplicate_and_renumber,
)
from deep_research_swarm.reporting.heatmap import render_confidence_heatmap
from deep_research_swarm.reporting.renderer import render_report


class TestRenderer:
    def _make_state(self, section_drafts, citations, gaps=None):
        return {
            "research_question": "What is quantum entanglement?",
            "section_drafts": section_drafts,
            "citations": citations,
            "research_gaps": gaps or [],
            "iteration_history": [
                IterationRecord(
                    iteration=1,
                    sub_queries_generated=3,
                    search_results_found=10,
                    documents_extracted=5,
                    sections_drafted=len(section_drafts),
                    avg_confidence=0.8,
                    sections_by_confidence={"HIGH": 1, "MEDIUM": 1, "LOW": 0},
                    token_usage=[],
                    replan_reason=None,
                )
            ],
            "total_cost_usd": 0.05,
            "total_tokens_used": 5000,
            "convergence_reason": "all_acceptable",
        }

    def test_yaml_frontmatter_present(self, sample_section_drafts, sample_citations):
        state = self._make_state(sample_section_drafts, sample_citations)
        report = render_report(state)
        assert report.startswith("---")
        # Extract and parse frontmatter
        parts = report.split("---", 2)
        assert len(parts) >= 3
        frontmatter = yaml.safe_load(parts[1])
        assert "title" in frontmatter
        assert frontmatter["total_citations"] == 2

    def test_confidence_table_present(self, sample_section_drafts, sample_citations):
        state = self._make_state(sample_section_drafts, sample_citations)
        report = render_report(state)
        assert "## Confidence Assessment" in report
        assert "| Section |" in report

    def test_citation_format(self, sample_section_drafts, sample_citations):
        state = self._make_state(sample_section_drafts, sample_citations)
        report = render_report(state)
        assert "## Bibliography" in report

    def test_gaps_section(self, sample_section_drafts, sample_citations):
        gaps = [
            ResearchGap(
                description="No data on long-term effects",
                attempted_queries=["long-term entanglement effects"],
                reason="no_sources",
            )
        ]
        state = self._make_state(sample_section_drafts, sample_citations, gaps)
        report = render_report(state)
        assert "## Research Gaps" in report
        assert "No data on long-term effects" in report


class TestCitations:
    def test_bibliography_format(self, sample_citations):
        bib = build_bibliography(sample_citations)
        assert "1." in bib
        assert "2." in bib
        assert "nature.com" in bib
        assert "arxiv.org" in bib

    def test_deduplicates_by_url(self):
        """Deduplication happens in deduplicate_and_renumber, not build_bibliography."""
        from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft

        sections = [
            SectionDraft(
                id="sec-001",
                heading="Test",
                content="Claim [1] and more [2].",
                citation_ids=["[1]", "[2]"],
                confidence_score=0.85,
                confidence_level=Confidence.HIGH,
                grader_scores=GraderScores(relevance=0.85, hallucination=0.9, quality=0.8),
            )
        ]
        cits = [
            Citation(
                id="[1]",
                url="https://example.com/a",
                title="A",
                authority=SourceAuthority.UNKNOWN,
                accessed="",
                used_in_sections=["Test"],
            ),
            Citation(
                id="[2]",
                url="https://example.com/a",  # Same URL
                title="A duplicate",
                authority=SourceAuthority.UNKNOWN,
                accessed="",
                used_in_sections=["Test"],
            ),
        ]
        deduped_sections, deduped_cits = deduplicate_and_renumber(sections, cits)
        bib = build_bibliography(deduped_cits)
        assert bib.count("example.com/a") == 1
        assert len(deduped_cits) == 1

    def test_empty_citations(self):
        assert build_bibliography([]) == ""


class TestDeduplicateAndRenumber:
    """V2: Full citation deduplication and renumbering pipeline."""

    def _make_section(self, sid, heading, content, cit_ids):
        from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft

        return SectionDraft(
            id=sid,
            heading=heading,
            content=content,
            citation_ids=cit_ids,
            confidence_score=0.85,
            confidence_level=Confidence.HIGH,
            grader_scores=GraderScores(relevance=0.85, hallucination=0.9, quality=0.8),
        )

    def _make_citation(self, cid, url, title):
        return Citation(
            id=cid,
            url=url,
            title=title,
            authority=SourceAuthority.INSTITUTIONAL,
            accessed="",
            used_in_sections=[],
        )

    def test_renumbers_sequentially(self):
        sections = [
            self._make_section("sec-1", "Intro", "Fact [1] and [2].", ["[1]", "[2]"]),
        ]
        cits = [
            self._make_citation("[1]", "https://a.com", "A"),
            self._make_citation("[2]", "https://b.com", "B"),
        ]
        new_secs, new_cits = deduplicate_and_renumber(sections, cits)
        assert new_cits[0]["id"] == "[1]"
        assert new_cits[1]["id"] == "[2]"
        assert "[1]" in new_secs[0]["content"]
        assert "[2]" in new_secs[0]["content"]

    def test_dedup_merges_duplicate_urls(self):
        sections = [
            self._make_section("sec-1", "Intro", "Fact [1] and [2].", ["[1]", "[2]"]),
        ]
        cits = [
            self._make_citation("[1]", "https://same.com", "Same A"),
            self._make_citation("[2]", "https://same.com", "Same B"),
        ]
        new_secs, new_cits = deduplicate_and_renumber(sections, cits)
        assert len(new_cits) == 1
        # Both old refs should now point to [1]
        assert new_secs[0]["content"].count("[1]") == 2
        assert "[2]" not in new_secs[0]["content"]

    def test_avoids_double_replacement_collision(self):
        """[10] should not get partially matched when renumbering [1]."""
        sections = [
            self._make_section(
                "sec-1",
                "Intro",
                "Cite [1] then [10].",
                ["[1]", "[10]"],
            ),
        ]
        cits = [
            self._make_citation("[1]", "https://a.com", "A"),
            self._make_citation("[10]", "https://b.com", "B"),
        ]
        new_secs, new_cits = deduplicate_and_renumber(sections, cits)
        assert len(new_cits) == 2
        content = new_secs[0]["content"]
        # Both citations should be present as [1] and [2]
        assert "[1]" in content
        assert "[2]" in content

    def test_empty_citations_passthrough(self):
        from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft

        sections = [
            SectionDraft(
                id="sec-1",
                heading="Intro",
                content="No citations here.",
                citation_ids=[],
                confidence_score=0.5,
                confidence_level=Confidence.MEDIUM,
                grader_scores=GraderScores(relevance=0.5, hallucination=1.0, quality=0.5),
            )
        ]
        new_secs, new_cits = deduplicate_and_renumber(sections, [])
        assert new_secs == sections
        assert new_cits == []

    def test_used_in_sections_merged(self):
        sections = [
            self._make_section("sec-1", "Intro", "Fact [1].", ["[1]"]),
            self._make_section("sec-2", "Body", "More [2].", ["[2]"]),
        ]
        cits = [
            self._make_citation("[1]", "https://same.com", "Same"),
            self._make_citation("[2]", "https://same.com", "Same Dup"),
        ]
        _, new_cits = deduplicate_and_renumber(sections, cits)
        assert len(new_cits) == 1
        assert "Intro" in new_cits[0]["used_in_sections"]
        assert "Body" in new_cits[0]["used_in_sections"]


class TestHeatmap:
    def test_renders_table(self, sample_section_drafts):
        table = render_confidence_heatmap(sample_section_drafts)
        assert "| Section |" in table
        assert "HIGH" in table
        assert "MEDIUM" in table

    def test_empty_sections(self):
        result = render_confidence_heatmap([])
        assert "No sections" in result

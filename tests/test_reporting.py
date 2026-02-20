"""Tests for report rendering."""

import yaml

from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    GraderScores,
    IterationRecord,
    ResearchGap,
    SectionDraft,
    SourceAuthority,
)
from deep_research_swarm.reporting.citations import build_bibliography
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
        cits = [
            Citation(
                id="[1]",
                url="https://example.com/a",
                title="A",
                authority=SourceAuthority.UNKNOWN,
                accessed="",
                used_in_sections=[],
            ),
            Citation(
                id="[2]",
                url="https://example.com/a",  # Same URL
                title="A duplicate",
                authority=SourceAuthority.UNKNOWN,
                accessed="",
                used_in_sections=[],
            ),
        ]
        bib = build_bibliography(cits)
        assert bib.count("example.com/a") == 1

    def test_empty_citations(self):
        assert build_bibliography([]) == ""


class TestHeatmap:
    def test_renders_table(self, sample_section_drafts):
        table = render_confidence_heatmap(sample_section_drafts)
        assert "| Section |" in table
        assert "HIGH" in table
        assert "MEDIUM" in table

    def test_empty_sections(self):
        result = render_confidence_heatmap([])
        assert "No sections" in result

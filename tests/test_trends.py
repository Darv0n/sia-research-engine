"""Tests for confidence trend visualization."""

from __future__ import annotations

from deep_research_swarm.contracts import IterationRecord, SectionConfidenceSnapshot
from deep_research_swarm.reporting.trends import _sparkline, render_confidence_trends


def _record(iteration: int, snapshots: list[SectionConfidenceSnapshot]) -> IterationRecord:
    """Helper to build a minimal IterationRecord."""
    return IterationRecord(
        iteration=iteration,
        sub_queries_generated=3,
        search_results_found=10,
        documents_extracted=8,
        sections_drafted=len(snapshots),
        avg_confidence=0.8,
        sections_by_confidence={"HIGH": len(snapshots)},
        token_usage=[],
        replan_reason=None,
        section_snapshots=snapshots,
    )


def _snap(heading: str, score: float) -> SectionConfidenceSnapshot:
    return SectionConfidenceSnapshot(
        heading=heading,
        confidence_score=score,
        confidence_level="HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.6 else "LOW",
    )


class TestConfidenceTrends:
    def test_single_iteration_empty(self):
        """Less than 2 iterations returns empty string."""
        history = [_record(1, [_snap("Intro", 0.85)])]
        assert render_confidence_trends(history) == ""

    def test_two_iterations_table(self):
        """Two iterations produce a valid Markdown table."""
        history = [
            _record(1, [_snap("Intro", 0.70), _snap("Methods", 0.60)]),
            _record(2, [_snap("Intro", 0.85), _snap("Methods", 0.75)]),
        ]
        result = render_confidence_trends(history)
        assert "| Section |" in result
        assert "Iter 1" in result
        assert "Iter 2" in result
        assert "Intro" in result
        assert "Methods" in result
        assert "0.70" in result
        assert "0.85" in result

    def test_sparkline_values(self):
        """Sparkline renders increasing values with ascending characters."""
        spark = _sparkline([0.0, 0.5, 1.0])
        assert len(spark) == 3
        # First char should be the lowest, last the highest
        assert spark[0] == "▁"
        assert spark[2] == "█"

    def test_new_section_in_later_iteration(self):
        """A section appearing only in iteration 2 shows a dash for iteration 1."""
        history = [
            _record(1, [_snap("Intro", 0.80)]),
            _record(2, [_snap("Intro", 0.90), _snap("Conclusion", 0.70)]),
        ]
        result = render_confidence_trends(history)
        assert "Conclusion" in result
        # Conclusion should have a dash for iter 1
        lines = result.split("\n")
        conclusion_line = [ln for ln in lines if "Conclusion" in ln][0]
        assert "—" in conclusion_line

    def test_dropped_section(self):
        """A section present in iter 1 but not iter 2 shows a dash for iter 2."""
        history = [
            _record(1, [_snap("Intro", 0.80), _snap("Legacy", 0.50)]),
            _record(2, [_snap("Intro", 0.90)]),
        ]
        result = render_confidence_trends(history)
        lines = result.split("\n")
        legacy_line = [ln for ln in lines if "Legacy" in ln][0]
        assert "0.50" in legacy_line
        assert "—" in legacy_line

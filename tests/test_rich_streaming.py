"""Tests for rich streaming events (V9)."""

from __future__ import annotations

from deep_research_swarm.streaming import StreamDisplay


class TestPlanSummaryEvent:
    def test_plan_summary_renders(self, capsys):
        display = StreamDisplay(verbose=False)
        display.handle_custom(
            {
                "kind": "plan_summary",
                "iteration": 1,
                "perspectives": ["technical", "economic"],
                "queries": ["query 1", "query 2", "query 3"],
            }
        )
        captured = capsys.readouterr()
        assert "Plan (iteration 1)" in captured.err
        assert "technical" in captured.err
        assert "query 1" in captured.err
        assert "Queries (3)" in captured.err


class TestFindingsPreviewEvent:
    def test_findings_preview_renders(self, capsys):
        display = StreamDisplay(verbose=False)
        display.handle_custom(
            {
                "kind": "findings_preview",
                "top_sources": ["Source 1 (url1)", "Source 2 (url2)"],
            }
        )
        captured = capsys.readouterr()
        assert "Top sources (2)" in captured.err
        assert "Source 1" in captured.err

    def test_findings_preview_limits_to_5(self, capsys):
        display = StreamDisplay(verbose=False)
        display.handle_custom(
            {
                "kind": "findings_preview",
                "top_sources": [f"Source {i}" for i in range(10)],
            }
        )
        captured = capsys.readouterr()
        # Should only show 5
        assert "Source 4" in captured.err
        assert "Source 5" not in captured.err


class TestContradictionSummaryEvent:
    def test_contradiction_summary_renders(self, capsys):
        display = StreamDisplay(verbose=False)
        display.handle_custom(
            {
                "kind": "contradiction_summary",
                "count": 3,
            }
        )
        captured = capsys.readouterr()
        assert "3 detected" in captured.err

    def test_zero_contradictions_silent(self, capsys):
        display = StreamDisplay(verbose=False)
        display.handle_custom(
            {
                "kind": "contradiction_summary",
                "count": 0,
            }
        )
        captured = capsys.readouterr()
        assert captured.err == ""


class TestGroundingSummaryEvent:
    def test_grounding_summary_renders(self, capsys):
        display = StreamDisplay(verbose=False)
        display.handle_custom(
            {
                "kind": "grounding_summary",
                "avg_score": 0.85,
                "total_passages": 42,
            }
        )
        captured = capsys.readouterr()
        assert "0.85" in captured.err
        assert "42" in captured.err


class TestNewNodeLabels:
    def test_gap_analysis_label(self):
        from deep_research_swarm.streaming import NODE_LABELS

        assert "gap_analysis" in NODE_LABELS
        assert "search_followup" in NODE_LABELS
        assert "extract_followup" in NODE_LABELS
        assert "score_merge" in NODE_LABELS
        assert "clarify" in NODE_LABELS

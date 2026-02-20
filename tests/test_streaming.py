"""Tests for streaming display."""

from __future__ import annotations

from deep_research_swarm.streaming import NODE_LABELS, StreamDisplay


class TestNodeLabels:
    def test_all_core_nodes_have_labels(self):
        """All core graph nodes have human-readable labels."""
        core_nodes = [
            "health_check",
            "plan",
            "search",
            "extract",
            "score",
            "synthesize",
            "critique",
            "rollup_budget",
            "report",
        ]
        for node in core_nodes:
            assert node in NODE_LABELS

    def test_labels_are_non_empty_strings(self):
        """All labels are non-empty strings."""
        for label in NODE_LABELS.values():
            assert isinstance(label, str) and len(label) > 0


class TestStreamDisplay:
    def test_iteration_detection(self):
        """StreamDisplay detects iteration from state delta."""
        display = StreamDisplay()
        state_delta = {"current_iteration": 2, "sub_queries": []}
        result = display._detect_iteration(state_delta)
        assert result == 2

    def test_iteration_detection_missing(self):
        """Returns None when current_iteration not in delta."""
        display = StreamDisplay()
        result = display._detect_iteration({"search_results": []})
        assert result is None

    def test_handle_update_sets_current_node(self, capsys):
        """handle_update tracks the current node."""
        display = StreamDisplay()
        display.handle_update({"search": {"search_results": []}})
        assert display._current_node == "search"

    def test_count_extraction_in_custom_event(self, capsys):
        """Custom event with count renders count in output."""
        display = StreamDisplay()
        display.handle_custom(
            {
                "kind": "search_progress",
                "message": "querying exa",
                "count": 15,
            }
        )
        captured = capsys.readouterr()
        assert "15 results" in captured.err
        assert "querying exa" in captured.err

    def test_verbose_mode_prints_details(self, capsys):
        """Verbose mode prints node output counts."""
        display = StreamDisplay(verbose=True)
        display.handle_update({"search": {"search_results": [{"url": "http://a.com"}] * 5}})
        captured = capsys.readouterr()
        assert "results=5" in captured.err

    def test_custom_event_unknown_kind(self, capsys):
        """Unknown custom event kind still prints the message."""
        display = StreamDisplay()
        display.handle_custom({"kind": "unknown", "message": "hello"})
        captured = capsys.readouterr()
        assert "hello" in captured.err

"""Tests for V5 CLI flags — --dump-state, --list-memories, --export-mcp, --no-memory."""

from __future__ import annotations

from deep_research_swarm.__main__ import _format_memory_context, parse_args


class TestNewCliFlags:
    def test_list_memories_flag(self):
        """--list-memories is a valid standalone flag."""
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["prog", "--list-memories"]):
            args = parse_args()
            assert args.list_memories is True

    def test_export_mcp_flag(self):
        """--export-mcp is a valid standalone flag."""
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["prog", "--export-mcp"]):
            args = parse_args()
            assert args.export_mcp is True

    def test_dump_state_flag(self):
        """--dump-state accepts a thread ID."""
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["prog", "--dump-state", "research-123"]):
            args = parse_args()
            assert args.dump_state == "research-123"

    def test_no_memory_flag(self):
        """--no-memory is parsed correctly with a question."""
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["prog", "--no-memory", "test question"]):
            args = parse_args()
            assert args.no_memory is True
            assert args.question == "test question"

    def test_no_memory_default_false(self):
        """--no-memory defaults to False."""
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["prog", "test question"]):
            args = parse_args()
            assert args.no_memory is False


class TestFormatMemoryContext:
    def test_empty_memories(self):
        assert _format_memory_context([]) == ""

    def test_single_memory_with_findings(self):
        memories = [
            {
                "question": "What is quantum computing?",
                "key_findings": ["Qubits", "Quantum Gates"],
                "gaps": ["Error correction"],
            }
        ]
        ctx = _format_memory_context(memories)
        assert "What is quantum computing?" in ctx
        assert "Qubits, Quantum Gates" in ctx
        assert "Error correction" in ctx

    def test_memory_without_findings_or_gaps(self):
        memories = [{"question": "Simple question", "key_findings": [], "gaps": []}]
        ctx = _format_memory_context(memories)
        assert "Simple question" in ctx
        assert "Findings" not in ctx
        assert "Open gaps" not in ctx

    def test_multiple_memories(self):
        memories = [
            {"question": "Q1", "key_findings": ["F1"], "gaps": []},
            {"question": "Q2", "key_findings": [], "gaps": ["G1"]},
        ]
        ctx = _format_memory_context(memories)
        assert "Q1" in ctx
        assert "Q2" in ctx

"""Tests for V4 resume CLI — arg parsing, resume flow, list-threads."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from deep_research_swarm.__main__ import parse_args


class TestArgParsing:
    def test_question_is_optional(self):
        """question arg is optional when --resume or --list-threads provided."""
        with patch.object(sys, "argv", ["prog", "--list-threads"]):
            args = parse_args()
            assert args.question is None
            assert args.list_threads is True

    def test_resume_flag_parsed(self):
        """--resume captures thread ID."""
        with patch.object(sys, "argv", ["prog", "--resume", "research-20260220-120000-ab12"]):
            args = parse_args()
            assert args.resume == "research-20260220-120000-ab12"
            assert args.question is None

    def test_question_still_works(self):
        """Positional question still works as before."""
        with patch.object(sys, "argv", ["prog", "What is quantum computing?"]):
            args = parse_args()
            assert args.question == "What is quantum computing?"
            assert args.resume is None
            assert args.list_threads is False

    def test_no_args_exits(self):
        """No question, --resume, or --list-threads causes error."""
        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_question_with_flags(self):
        """Question works alongside other flags."""
        with patch.object(sys, "argv", ["prog", "test q", "--max-iterations", "2", "--no-cache"]):
            args = parse_args()
            assert args.question == "test q"
            assert args.max_iterations == 2
            assert args.no_cache is True


class TestResumeValidation:
    @patch("deep_research_swarm.__main__.get_settings")
    @pytest.mark.asyncio
    async def test_resume_with_none_backend_errors(self, mock_settings):
        """Resume with checkpoint_backend='none' prints error."""
        from deep_research_swarm.__main__ import run

        settings = MagicMock()
        settings.validate.return_value = []
        settings.checkpoint_backend = "none"
        settings.checkpoint_db = "checkpoints/research.db"
        mock_settings.return_value = settings

        args = MagicMock()
        args.resume = "research-123"
        args.list_threads = False
        args.question = None

        with pytest.raises(SystemExit):
            await run(args)

    @patch("deep_research_swarm.__main__.get_settings")
    @pytest.mark.asyncio
    async def test_resume_missing_db_errors(self, mock_settings):
        """Resume when DB file doesn't exist prints error."""
        from deep_research_swarm.__main__ import run

        settings = MagicMock()
        settings.validate.return_value = []
        settings.checkpoint_backend = "sqlite"
        settings.checkpoint_db = "nonexistent/path/research.db"
        mock_settings.return_value = settings

        args = MagicMock()
        args.resume = "research-123"
        args.list_threads = False
        args.question = None
        args.no_cache = False

        with pytest.raises(SystemExit):
            await run(args)


class TestOutputReport:
    def test_output_report_no_report_exits(self):
        """_output_report exits when no report in result."""
        from deep_research_swarm.__main__ import _output_report

        with pytest.raises(SystemExit):
            _output_report({}, output_path_override=None, question="test")

    def test_output_report_writes_file(self, tmp_path):
        """_output_report writes report to specified path."""
        from deep_research_swarm.__main__ import _output_report

        out_file = tmp_path / "report.md"
        result = {
            "final_report": "# Test Report\nContent here.",
            "total_tokens_used": 1000,
            "total_cost_usd": 0.05,
            "iteration_history": [{}],
            "convergence_reason": "budget",
        }
        _output_report(result, output_path_override=str(out_file), question="test q")
        assert out_file.read_text(encoding="utf-8") == "# Test Report\nContent here."

"""Tests for multi-format export (V9)."""

from __future__ import annotations

from unittest.mock import patch

from deep_research_swarm.reporting.export import convert_report, pandoc_available


class TestPandocAvailable:
    def test_returns_bool(self):
        result = pandoc_available()
        assert isinstance(result, bool)

    def test_false_when_not_installed(self):
        with patch("deep_research_swarm.reporting.export.shutil.which", return_value=None):
            assert pandoc_available() is False

    def test_true_when_installed(self):
        with patch(
            "deep_research_swarm.reporting.export.shutil.which",
            return_value="/usr/bin/pandoc",
        ):
            assert pandoc_available() is True


class TestConvertReport:
    def test_unsupported_format(self):
        assert convert_report("# Test", "html", "out.html") is False

    def test_fails_gracefully_without_pandoc(self):
        with patch("deep_research_swarm.reporting.export.pandoc_available", return_value=False):
            assert convert_report("# Test", "docx", "out.docx") is False

    def test_calls_pandoc_for_docx(self):
        with (
            patch("deep_research_swarm.reporting.export.pandoc_available", return_value=True),
            patch("deep_research_swarm.reporting.export.subprocess.run") as mock_run,
            patch("deep_research_swarm.reporting.export.tempfile.NamedTemporaryFile") as mock_tmp,
        ):
            # Mock temp file
            mock_tmp.return_value.__enter__ = lambda s: s
            mock_tmp.return_value.__exit__ = lambda s, *a: None
            mock_tmp.return_value.name = "/tmp/test.md"
            mock_tmp.return_value.write = lambda x: None

            # Mock successful run
            mock_run.return_value.returncode = 0

            with patch("deep_research_swarm.reporting.export.Path.unlink"):
                result = convert_report("# Test Report", "docx", "output.docx")

            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "pandoc" in args
            assert "output.docx" in args

    def test_handles_pandoc_failure(self):
        with (
            patch("deep_research_swarm.reporting.export.pandoc_available", return_value=True),
            patch("deep_research_swarm.reporting.export.subprocess.run") as mock_run,
            patch("deep_research_swarm.reporting.export.tempfile.NamedTemporaryFile") as mock_tmp,
        ):
            mock_tmp.return_value.__enter__ = lambda s: s
            mock_tmp.return_value.__exit__ = lambda s, *a: None
            mock_tmp.return_value.name = "/tmp/test.md"
            mock_tmp.return_value.write = lambda x: None

            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Conversion error"

            with patch("deep_research_swarm.reporting.export.Path.unlink"):
                result = convert_report("# Test", "docx", "output.docx")

            assert result is False


class TestFormatCliFlag:
    def test_format_flag_in_argparse(self):
        import sys

        from deep_research_swarm.__main__ import parse_args

        with patch.object(sys, "argv", ["prog", "test question", "--format", "docx"]):
            args = parse_args()
            assert args.format == "docx"

    def test_format_default_is_md(self):
        import sys

        from deep_research_swarm.__main__ import parse_args

        with patch.object(sys, "argv", ["prog", "test question"]):
            args = parse_args()
            assert args.format == "md"

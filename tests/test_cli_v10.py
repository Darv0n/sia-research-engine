"""Tests for V10 CLI flags: --swarm and --no-sia."""

from __future__ import annotations

import sys
from unittest.mock import patch

from deep_research_swarm.__main__ import parse_args


class TestSwarmFlag:
    """--swarm N CLI flag."""

    def test_swarm_default_disabled(self):
        with patch.object(sys, "argv", ["prog", "test question"]):
            args = parse_args()
            assert args.swarm == 0

    def test_swarm_with_value(self):
        with patch.object(sys, "argv", ["prog", "test question", "--swarm", "3"]):
            args = parse_args()
            assert args.swarm == 3

    def test_swarm_value_2(self):
        with patch.object(sys, "argv", ["prog", "test question", "--swarm", "2"]):
            args = parse_args()
            assert args.swarm == 2

    def test_swarm_value_5(self):
        with patch.object(sys, "argv", ["prog", "test question", "--swarm", "5"]):
            args = parse_args()
            assert args.swarm == 5


class TestNoSiaFlag:
    """--no-sia CLI flag."""

    def test_no_sia_default_false(self):
        with patch.object(sys, "argv", ["prog", "test question"]):
            args = parse_args()
            assert args.no_sia is False

    def test_no_sia_when_set(self):
        with patch.object(sys, "argv", ["prog", "test question", "--no-sia"]):
            args = parse_args()
            assert args.no_sia is True


class TestV10FlagCombinations:
    """Combined V10 flags with other flags."""

    def test_swarm_with_verbose(self):
        with patch.object(sys, "argv", ["prog", "test question", "--swarm", "3", "--verbose"]):
            args = parse_args()
            assert args.swarm == 3
            assert args.verbose is True

    def test_no_sia_with_no_adaptive(self):
        with patch.object(sys, "argv", ["prog", "test question", "--no-sia", "--no-adaptive"]):
            args = parse_args()
            assert args.no_sia is True
            assert args.no_adaptive is True

    def test_swarm_with_no_sia(self):
        with patch.object(sys, "argv", ["prog", "test question", "--swarm", "3", "--no-sia"]):
            args = parse_args()
            assert args.swarm == 3
            assert args.no_sia is True

    def test_swarm_with_format(self):
        with patch.object(
            sys, "argv", ["prog", "test question", "--swarm", "2", "--format", "pdf"]
        ):
            args = parse_args()
            assert args.swarm == 2
            assert args.format == "pdf"


class TestSwarmEnabledConfig:
    """C9 regression: SWARM_ENABLED config must be honored in runtime logic."""

    def test_swarm_enabled_field_exists(self):
        from deep_research_swarm.config import Settings

        s = Settings()
        assert hasattr(s, "swarm_enabled")

    def test_swarm_enabled_default_true(self):
        from deep_research_swarm.config import Settings

        s = Settings()
        assert s.swarm_enabled is True

    def test_swarm_enabled_false_override(self):
        from deep_research_swarm.config import Settings

        s = Settings(swarm_enabled=False)
        assert s.swarm_enabled is False

    def test_swarm_disabled_check_in_main(self):
        """The __main__.py must check swarm_enabled before running swarm."""
        import inspect

        from deep_research_swarm.__main__ import run

        source = inspect.getsource(run)
        assert "swarm_enabled" in source, (
            "run() must check settings.swarm_enabled to honor SWARM_ENABLED env var"
        )

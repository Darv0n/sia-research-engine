"""Tests for follow-up questions CLI flag (V9)."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


class TestFollowUpArgparse:
    def test_follow_up_flag_parsing(self):
        from deep_research_swarm.__main__ import parse_args

        with patch.object(
            sys,
            "argv",
            ["prog", "--follow-up", "research-20260223-120000-abcd", "What about X?"],
        ):
            args = parse_args()
            assert args.follow_up == [
                "research-20260223-120000-abcd",
                "What about X?",
            ]

    def test_follow_up_requires_two_args(self):
        from deep_research_swarm.__main__ import parse_args

        with patch.object(
            sys,
            "argv",
            ["prog", "--follow-up", "thread-id-only"],
        ):
            with pytest.raises(SystemExit):
                parse_args()

    def test_no_follow_up_by_default(self):
        from deep_research_swarm.__main__ import parse_args

        with patch.object(sys, "argv", ["prog", "test question"]):
            args = parse_args()
            assert args.follow_up is None

    def test_follow_up_doesnt_require_question(self):
        """--follow-up provides its own question, no positional arg needed."""
        from deep_research_swarm.__main__ import parse_args

        with patch.object(
            sys,
            "argv",
            ["prog", "--follow-up", "thread-id", "follow up question"],
        ):
            args = parse_args()
            assert args.question is None
            assert args.follow_up is not None

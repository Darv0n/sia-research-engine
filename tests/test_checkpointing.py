"""Tests for V4 checkpointing — config, build_graph integration, thread ID format."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from deep_research_swarm.config import Settings


class TestCheckpointConfig:
    def test_default_checkpoint_db(self):
        """Default checkpoint_db is checkpoints/research.db."""
        s = Settings(anthropic_api_key="k")
        assert s.checkpoint_db == "checkpoints/research.db"

    def test_default_checkpoint_backend(self):
        """Default checkpoint_backend is sqlite."""
        s = Settings(anthropic_api_key="k")
        assert s.checkpoint_backend == "sqlite"

    def test_none_backend_valid(self):
        """checkpoint_backend='none' passes validation."""
        s = Settings(anthropic_api_key="k", checkpoint_backend="none")
        errors = s.validate()
        assert not any("CHECKPOINT_BACKEND" in e for e in errors)

    def test_sqlite_backend_valid(self):
        """checkpoint_backend='sqlite' passes validation."""
        s = Settings(anthropic_api_key="k", checkpoint_backend="sqlite")
        errors = s.validate()
        assert not any("CHECKPOINT_BACKEND" in e for e in errors)

    def test_postgres_valid_with_dsn(self):
        """checkpoint_backend='postgres' passes validation when POSTGRES_DSN is set."""
        s = Settings(
            anthropic_api_key="k",
            checkpoint_backend="postgres",
            postgres_dsn="postgresql://user:pass@localhost/db",
        )
        errors = s.validate()
        assert not any("CHECKPOINT_BACKEND" in e for e in errors)
        assert not any("POSTGRES_DSN" in e for e in errors)

    def test_postgres_invalid_without_dsn(self):
        """checkpoint_backend='postgres' fails validation without POSTGRES_DSN."""
        s = Settings(anthropic_api_key="k", checkpoint_backend="postgres", postgres_dsn="")
        errors = s.validate()
        assert any("POSTGRES_DSN" in e for e in errors)

    def test_unknown_backend_rejected(self):
        """Unknown checkpoint_backend is rejected."""
        s = Settings(anthropic_api_key="k", checkpoint_backend="redis")
        errors = s.validate()
        assert any("CHECKPOINT_BACKEND" in e for e in errors)

    def test_custom_db_path(self):
        """Checkpoint DB path can be customized."""
        s = Settings(anthropic_api_key="k", checkpoint_db="/tmp/custom.db")
        assert s.checkpoint_db == "/tmp/custom.db"


class TestBuildGraphWithCheckpointer:
    @patch("deep_research_swarm.graph.builder.AgentCaller")
    def test_compile_with_checkpointer(self, mock_caller_cls):
        """build_graph passes checkpointer to compile()."""
        from langgraph.checkpoint.memory import InMemorySaver

        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(
            anthropic_api_key="test-key",
            searxng_url="http://localhost:8080",
        )
        mock_caller_cls.return_value = MagicMock()

        checkpointer = InMemorySaver()
        graph = build_graph(settings, checkpointer=checkpointer)
        assert graph is not None

    @patch("deep_research_swarm.graph.builder.AgentCaller")
    def test_compile_without_checkpointer(self, mock_caller_cls):
        """build_graph works without checkpointer (V3 compat)."""
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(
            anthropic_api_key="test-key",
            searxng_url="http://localhost:8080",
        )
        mock_caller_cls.return_value = MagicMock()

        graph = build_graph(settings)
        assert graph is not None


class TestThreadIdFormat:
    def test_format_matches_pattern(self):
        """Thread ID matches research-YYYYMMDD-HHMMSS-XXXX pattern."""
        from deep_research_swarm.__main__ import _generate_thread_id

        tid = _generate_thread_id()
        assert re.match(r"^research-\d{8}-\d{6}-[0-9a-f]{4}$", tid)

    def test_unique_ids(self):
        """Multiple calls produce different IDs."""
        from deep_research_swarm.__main__ import _generate_thread_id

        ids = {_generate_thread_id() for _ in range(10)}
        assert len(ids) == 10

"""Tests for checkpoint robustness — WAL mode, large state writes, no-checkpointer fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from deep_research_swarm.config import Settings


class TestSqliteWALMode:
    """Verify that _make_checkpointer sets WAL mode and busy_timeout."""

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path):
        """SQLite checkpointer uses WAL journal mode."""
        from deep_research_swarm.__main__ import _make_checkpointer

        db_path = str(tmp_path / "test.db")
        settings = Settings(anthropic_api_key="k", checkpoint_backend="sqlite")

        async with _make_checkpointer(settings, db_path) as checkpointer:
            # The underlying connection should have WAL mode
            cursor = await checkpointer.conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] == "wal"

    @pytest.mark.asyncio
    async def test_busy_timeout_set(self, tmp_path):
        """SQLite checkpointer has generous busy_timeout."""
        from deep_research_swarm.__main__ import _make_checkpointer

        db_path = str(tmp_path / "test.db")
        settings = Settings(anthropic_api_key="k", checkpoint_backend="sqlite")

        async with _make_checkpointer(settings, db_path) as checkpointer:
            cursor = await checkpointer.conn.execute("PRAGMA busy_timeout")
            row = await cursor.fetchone()
            assert row[0] >= 30000

    @pytest.mark.asyncio
    async def test_none_backend_yields_none(self):
        """checkpoint_backend='none' yields None checkpointer."""
        from deep_research_swarm.__main__ import _make_checkpointer

        settings = Settings(anthropic_api_key="k", checkpoint_backend="none")

        async with _make_checkpointer(settings, "unused.db") as checkpointer:
            assert checkpointer is None


class TestLargeStateCheckpoint:
    """Simulate large state writes to verify SQLite doesn't lock."""

    @pytest.mark.asyncio
    async def test_large_state_write(self, tmp_path):
        """Checkpointer handles large state (simulating 1000+ scored docs)."""
        from deep_research_swarm.__main__ import _make_checkpointer

        db_path = str(tmp_path / "large_state.db")
        settings = Settings(anthropic_api_key="k", checkpoint_backend="sqlite")

        async with _make_checkpointer(settings, db_path) as checkpointer:
            await checkpointer.setup()

            # Simulate a large state payload (~1000 scored documents)
            large_docs = [
                {
                    "url": f"https://example.com/doc-{i}",
                    "score": 0.5 + (i % 50) / 100,
                    "text": f"Content for document {i} " * 50,
                }
                for i in range(1000)
            ]
            config = {"configurable": {"thread_id": "test-large-state", "checkpoint_ns": ""}}

            # Write large checkpoint
            import json

            checkpoint = {
                "v": 1,
                "id": "checkpoint-1",
                "ts": "2026-02-23T00:00:00Z",
                "channel_values": {
                    "scored_documents": json.dumps(large_docs),
                },
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            await checkpointer.aput(config, checkpoint, {}, {})

            # Verify we can read it back
            got = await checkpointer.aget(config)
            assert got is not None

    @pytest.mark.asyncio
    async def test_concurrent_writes_no_lock(self, tmp_path):
        """Multiple concurrent writes don't trigger database locked error."""
        from deep_research_swarm.__main__ import _make_checkpointer

        db_path = str(tmp_path / "concurrent.db")
        settings = Settings(anthropic_api_key="k", checkpoint_backend="sqlite")

        async with _make_checkpointer(settings, db_path) as checkpointer:
            await checkpointer.setup()

            import json

            async def write_checkpoint(thread_id: str, size: int):
                docs = [
                    {"url": f"https://example.com/{thread_id}/{i}", "text": "x" * 200}
                    for i in range(size)
                ]
                config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                checkpoint = {
                    "v": 1,
                    "id": f"cp-{thread_id}",
                    "ts": "2026-02-23T00:00:00Z",
                    "channel_values": {"data": json.dumps(docs)},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                }
                await checkpointer.aput(config, checkpoint, {}, {})

            # Fire multiple concurrent writes
            tasks = [write_checkpoint(f"thread-{i}", 200) for i in range(5)]
            # Should complete without sqlite3.OperationalError
            await asyncio.gather(*tasks)


class TestHasCheckpointer:
    """Test _has_checkpointer helper and fallback behavior."""

    def test_has_checkpointer_true(self):
        from deep_research_swarm.__main__ import _has_checkpointer

        graph = MagicMock()
        graph.checkpointer = MagicMock()
        assert _has_checkpointer(graph) is True

    def test_has_checkpointer_false_none(self):
        from deep_research_swarm.__main__ import _has_checkpointer

        graph = MagicMock()
        graph.checkpointer = None
        assert _has_checkpointer(graph) is False

    def test_has_checkpointer_false_missing(self):
        from deep_research_swarm.__main__ import _has_checkpointer

        graph = MagicMock(spec=[])  # no attributes
        assert _has_checkpointer(graph) is False

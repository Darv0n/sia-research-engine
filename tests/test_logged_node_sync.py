"""Tests for _wrap_with_logging handling sync and async node functions."""

from __future__ import annotations

import pytest

from deep_research_swarm.event_log.writer import EventLog
from deep_research_swarm.graph.builder import _wrap_with_logging


def _make_event_log(tmp_path) -> EventLog:
    return EventLog(tmp_path, "test-thread")


# --- Sync node (like adapt_extraction_node / adapt_synthesis_node) ---


def _sync_node(state: dict) -> dict:
    """Mimics adapt_extraction_node — sync, single param."""
    return {"tunable_snapshot": {"extraction_cap": 60}, "adaptation_events": []}


def _sync_node_with_config(state: dict, config=None) -> dict:
    """Hypothetical sync node that accepts config."""
    return {"tunable_snapshot": {"extraction_cap": 60}, "adaptation_events": []}


# --- Async node (like every other node) ---


async def _async_node(state: dict) -> dict:
    """Mimics a normal async node."""
    return {"sub_queries": [{"query": "test"}], "token_usage": []}


async def _async_node_with_config(state: dict, config=None) -> dict:
    """Mimics an async node that accepts config (e.g. score_node)."""
    return {"scored_documents": [], "token_usage": []}


# --- Tests ---


class TestSyncNodeWrapping:
    """Fix: sync nodes must not be awaited by the logging wrapper."""

    @pytest.mark.asyncio
    async def test_sync_node_returns_result(self, tmp_path):
        """Sync node wrapped with logging should return its dict without error."""
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("adapt_extraction", _sync_node, event_log)
        result = await wrapped({"current_iteration": 1})
        assert result["tunable_snapshot"]["extraction_cap"] == 60
        assert result["adaptation_events"] == []

    @pytest.mark.asyncio
    async def test_sync_node_emits_event(self, tmp_path):
        """Sync node should still produce a RunEvent via the wrapper."""
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("adapt_extraction", _sync_node, event_log)
        await wrapped({"current_iteration": 1})
        events = event_log.read_all()
        assert len(events) == 1
        assert events[0]["node"] == "adapt_extraction"

    @pytest.mark.asyncio
    async def test_sync_node_with_config_param(self, tmp_path):
        """Sync node with 2 params (state, config) should also work."""
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("adapt_extraction", _sync_node_with_config, event_log)
        result = await wrapped({"current_iteration": 1})
        assert result["tunable_snapshot"]["extraction_cap"] == 60


class TestAsyncNodeWrapping:
    """Verify async nodes still work correctly after the sync fix."""

    @pytest.mark.asyncio
    async def test_async_node_returns_result(self, tmp_path):
        """Async node wrapped with logging should return its dict."""
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("plan", _async_node, event_log)
        result = await wrapped({"current_iteration": 1})
        assert result["sub_queries"] == [{"query": "test"}]

    @pytest.mark.asyncio
    async def test_async_node_emits_event(self, tmp_path):
        """Async node should produce a RunEvent via the wrapper."""
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("plan", _async_node, event_log)
        await wrapped({"current_iteration": 1})
        events = event_log.read_all()
        assert len(events) == 1
        assert events[0]["node"] == "plan"

    @pytest.mark.asyncio
    async def test_async_node_with_config_param(self, tmp_path):
        """Async node with config param should work."""
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("score", _async_node_with_config, event_log)
        result = await wrapped({"current_iteration": 1})
        assert result["scored_documents"] == []


class TestEventLogAccuracy:
    """Verify the emitted event captures elapsed time and token stats."""

    @pytest.mark.asyncio
    async def test_elapsed_time_recorded(self, tmp_path):
        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("adapt_extraction", _sync_node, event_log)
        await wrapped({"current_iteration": 2})
        events = event_log.read_all()
        assert events[0]["elapsed_s"] >= 0
        assert events[0]["iteration"] == 2

    @pytest.mark.asyncio
    async def test_token_usage_from_node_result(self, tmp_path):
        """Nodes returning token_usage should have it aggregated in the event."""

        async def node_with_tokens(state):
            return {
                "token_usage": [
                    {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01},
                    {"input_tokens": 200, "output_tokens": 100, "cost_usd": 0.02},
                ]
            }

        event_log = _make_event_log(tmp_path)
        wrapped = _wrap_with_logging("synthesize", node_with_tokens, event_log)
        await wrapped({"current_iteration": 1})
        events = event_log.read_all()
        assert events[0]["tokens"] == 450  # 100+50+200+100
        assert events[0]["cost"] == 0.03

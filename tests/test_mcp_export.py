"""Tests for MCP entity export format."""

from __future__ import annotations

import json

from deep_research_swarm.memory.mcp_export import export_to_mcp_format


def _make_memory(**overrides) -> dict:
    base = {
        "thread_id": "research-20260220-120000-abcd",
        "question": "What is quantum computing?",
        "timestamp": "2026-02-20T12:00:00Z",
        "key_findings": ["Qubits", "Quantum Gates"],
        "gaps": ["Error correction"],
        "sources_count": 15,
        "iterations": 3,
        "converged": True,
    }
    base.update(overrides)
    return base


class TestExportToMcpFormat:
    def test_entity_structure(self):
        """Each memory produces an entity with required MCP fields."""
        result = json.loads(export_to_mcp_format([_make_memory()]))
        assert len(result) == 1
        entity = result[0]
        assert entity["name"] == "research-20260220-120000-abcd"
        assert entity["entityType"] == "ResearchRun"
        assert isinstance(entity["observations"], list)
        assert isinstance(entity["metadata"], dict)

    def test_observations_include_findings_and_gaps(self):
        """Observations contain Finding: and Gap: prefixed entries."""
        result = json.loads(export_to_mcp_format([_make_memory()]))
        obs = result[0]["observations"]
        assert "Finding: Qubits" in obs
        assert "Finding: Quantum Gates" in obs
        assert "Gap: Error correction" in obs

    def test_observations_include_stats(self):
        """Last observation contains source count, iterations, and convergence."""
        result = json.loads(export_to_mcp_format([_make_memory()]))
        stats = result[0]["observations"][-1]
        assert "15 sources" in stats
        assert "3 iterations" in stats
        assert "converged=True" in stats

    def test_metadata_fields(self):
        """Metadata contains question and timestamp."""
        result = json.loads(export_to_mcp_format([_make_memory()]))
        meta = result[0]["metadata"]
        assert meta["question"] == "What is quantum computing?"
        assert meta["timestamp"] == "2026-02-20T12:00:00Z"

    def test_empty_input(self):
        """Empty list produces empty JSON array."""
        result = json.loads(export_to_mcp_format([]))
        assert result == []

    def test_multiple_memories(self):
        """Multiple memories produce multiple entities."""
        memories = [
            _make_memory(thread_id="t1", question="Q1"),
            _make_memory(thread_id="t2", question="Q2"),
        ]
        result = json.loads(export_to_mcp_format(memories))
        assert len(result) == 2
        assert result[0]["name"] == "t1"
        assert result[1]["name"] == "t2"

    def test_no_findings_no_gaps(self):
        """Memory with empty findings and gaps still has stats observation."""
        mem = _make_memory(key_findings=[], gaps=[])
        result = json.loads(export_to_mcp_format([mem]))
        obs = result[0]["observations"]
        assert len(obs) == 1  # Just the stats line
        assert "Stats:" in obs[0]

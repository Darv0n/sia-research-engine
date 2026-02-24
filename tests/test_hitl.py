"""Tests for HITL mode — gate node insertion and edge wiring."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from deep_research_swarm.config import Settings
from deep_research_swarm.graph.builder import build_graph


def _make_settings(**overrides) -> Settings:
    defaults = {
        "anthropic_api_key": "test-key",
        "exa_api_key": "",
        "tavily_api_key": "",
        "searxng_url": "http://localhost:8080",
        "opus_model": "claude-opus-4-6",
        "sonnet_model": "claude-sonnet-4-6",
        "max_iterations": 3,
        "token_budget": 200000,
        "max_concurrent_requests": 5,
        "authority_weight": 0.2,
        "rrf_k": 60,
        "convergence_threshold": 0.05,
        "search_cache_ttl": 3600,
        "search_cache_dir": ".cache/search",
        "checkpoint_db": "checkpoints/test.db",
        "checkpoint_backend": "sqlite",
        "memory_dir": "memory/",
        "postgres_dsn": "",
        "run_log_dir": "runs/",
        "mode": "auto",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _get_node_names(graph) -> set[str]:
    """Extract node names from a compiled graph."""
    return set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}


def _get_edges(graph) -> list[tuple[str, str]]:
    """Extract edges as (source, target) from a compiled graph."""
    edges = []
    for edge in graph.get_graph().edges:
        edges.append((edge.source, edge.target))
    return edges


class TestAutoModeNoGates:
    def test_auto_mode_has_no_gate_nodes(self):
        settings = _make_settings()
        graph = build_graph(settings, mode="auto")
        nodes = _get_node_names(graph)
        assert "plan_gate" not in nodes
        assert "report_gate" not in nodes

    def test_auto_mode_with_checkpointer_still_no_gates(self):
        settings = _make_settings()
        checkpointer = MemorySaver()
        graph = build_graph(settings, checkpointer=checkpointer, mode="auto")
        nodes = _get_node_names(graph)
        assert "plan_gate" not in nodes
        assert "report_gate" not in nodes


class TestHitlModeWithoutCheckpointer:
    def test_hitl_without_checkpointer_no_gates(self):
        """HITL mode gracefully falls back to auto when no checkpointer."""
        settings = _make_settings()
        graph = build_graph(settings, mode="hitl")
        nodes = _get_node_names(graph)
        assert "plan_gate" not in nodes
        assert "report_gate" not in nodes


class TestHitlModeWithCheckpointer:
    def test_hitl_has_gate_nodes(self):
        settings = _make_settings()
        checkpointer = MemorySaver()
        graph = build_graph(settings, checkpointer=checkpointer, mode="hitl")
        nodes = _get_node_names(graph)
        assert "plan_gate" in nodes
        assert "report_gate" in nodes

    def test_plan_gate_wiring(self):
        """plan -> plan_gate -> search"""
        settings = _make_settings()
        checkpointer = MemorySaver()
        graph = build_graph(settings, checkpointer=checkpointer, mode="hitl")
        edges = _get_edges(graph)
        assert ("plan", "plan_gate") in edges
        assert ("plan_gate", "search") in edges
        # Direct plan->search should NOT exist
        assert ("plan", "search") not in edges

    def test_report_gate_wiring(self):
        """report -> report_gate -> __end__"""
        settings = _make_settings()
        checkpointer = MemorySaver()
        graph = build_graph(settings, checkpointer=checkpointer, mode="hitl")
        edges = _get_edges(graph)
        assert ("report", "report_gate") in edges
        assert ("report_gate", "__end__") in edges
        # Direct report->__end__ should NOT exist
        assert ("report", "__end__") not in edges

    def test_core_pipeline_edges_preserved(self):
        """Non-gate edges remain the same in HITL mode."""
        settings = _make_settings()
        checkpointer = MemorySaver()
        graph = build_graph(settings, checkpointer=checkpointer, mode="hitl")
        edges = _get_edges(graph)
        assert ("health_check", "plan") in edges
        assert ("search", "adapt_extraction") in edges
        assert ("adapt_extraction", "extract") in edges
        assert ("extract", "chunk_passages") in edges
        assert ("chunk_passages", "score") in edges
        assert ("score", "citation_chain") in edges
        assert ("citation_chain", "contradiction") in edges
        assert ("contradiction", "synthesize") in edges
        assert ("synthesize", "critique") in edges
        assert ("critique", "rollup_budget") in edges


class TestConfigModeValidation:
    def test_valid_auto_mode(self):
        settings = _make_settings(mode="auto")
        errors = settings.validate()
        mode_errors = [e for e in errors if "MODE" in e]
        assert mode_errors == []

    def test_valid_hitl_mode(self):
        settings = _make_settings(mode="hitl")
        errors = settings.validate()
        mode_errors = [e for e in errors if "MODE" in e]
        assert mode_errors == []

    def test_invalid_mode_rejected(self):
        settings = _make_settings(mode="invalid")
        errors = settings.validate()
        mode_errors = [e for e in errors if "MODE" in e]
        assert len(mode_errors) == 1

"""Tests for gap analyzer — reactive search within a single iteration (V9)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deep_research_swarm.agents.gap_analyzer import analyze_gaps


def _make_scored_doc(url: str = "https://example.com", title: str = "Test", **kwargs):
    return {
        "id": "doc-1",
        "url": url,
        "title": title,
        "content": kwargs.get("content", "Test content for analysis"),
        "rrf_score": 0.5,
        "authority": "unknown",
        "authority_score": 0.5,
        "combined_score": 0.5,
        "sub_query_ids": ["sq-1"],
    }


class TestAnalyzeGaps:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_scored_docs(self):
        state = {"research_question": "test question", "scored_documents": []}
        caller = AsyncMock()
        result = await analyze_gaps(state, caller)
        assert result["follow_up_queries"] == []

    @pytest.mark.asyncio
    async def test_skips_when_followup_already_done(self):
        state = {
            "research_question": "test question",
            "scored_documents": [_make_scored_doc()],
            "follow_up_round": 1,
        }
        caller = AsyncMock()
        result = await analyze_gaps(state, caller)
        assert result["follow_up_queries"] == []

    @pytest.mark.asyncio
    async def test_returns_followup_queries_on_gaps_found(self):
        state = {
            "research_question": "What is quantum computing?",
            "scored_documents": [_make_scored_doc()],
            "sub_queries": [{"id": "sq-1", "question": "quantum computing overview"}],
        }
        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(
                {
                    "gaps_found": True,
                    "gaps": [
                        {
                            "description": "Missing hardware details",
                            "query": "quantum computing hardware superconducting qubits",
                            "priority": 1,
                        },
                        {
                            "description": "No error correction coverage",
                            "query": "quantum error correction surface codes",
                            "priority": 2,
                        },
                    ],
                },
                {
                    "agent": "gap_analyzer",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        result = await analyze_gaps(state, mock_caller)
        assert len(result["follow_up_queries"]) == 2
        assert result["follow_up_round"] == 1
        assert result["follow_up_queries"][0]["perspective"] == "follow-up"
        assert result["follow_up_queries"][0]["id"].startswith("sq-fu-")

    @pytest.mark.asyncio
    async def test_no_gaps_found_returns_empty(self):
        state = {
            "research_question": "comprehensive topic",
            "scored_documents": [_make_scored_doc()],
            "sub_queries": [],
        }
        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(
                {"gaps_found": False, "gaps": []},
                {
                    "agent": "gap_analyzer",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        result = await analyze_gaps(state, mock_caller)
        assert result["follow_up_queries"] == []
        assert result["follow_up_round"] == 1

    @pytest.mark.asyncio
    async def test_deduplicates_against_existing_queries(self):
        state = {
            "research_question": "test",
            "scored_documents": [_make_scored_doc()],
            "sub_queries": [{"id": "sq-1", "question": "exact duplicate query"}],
        }
        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(
                {
                    "gaps_found": True,
                    "gaps": [
                        {"description": "dup", "query": "exact duplicate query", "priority": 1},
                        {"description": "new", "query": "brand new unique query", "priority": 2},
                    ],
                },
                {
                    "agent": "gap_analyzer",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        result = await analyze_gaps(state, mock_caller)
        # Duplicate should be filtered, only 1 query should remain
        assert len(result["follow_up_queries"]) == 1
        assert "brand new unique query" in result["follow_up_queries"][0]["question"]

    @pytest.mark.asyncio
    async def test_respects_follow_up_budget_tunable(self):
        state = {
            "research_question": "test",
            "scored_documents": [_make_scored_doc()],
            "sub_queries": [],
            "tunable_snapshot": {"follow_up_budget": 2},
        }
        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(
                {
                    "gaps_found": True,
                    "gaps": [
                        {"description": f"gap {i}", "query": f"query {i}", "priority": i}
                        for i in range(5)
                    ],
                },
                {
                    "agent": "gap_analyzer",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        result = await analyze_gaps(state, mock_caller)
        assert len(result["follow_up_queries"]) == 2  # budget = 2

    @pytest.mark.asyncio
    async def test_adds_to_sub_queries_for_history(self):
        """Follow-up queries should also be added to sub_queries for dedup across iterations."""
        state = {
            "research_question": "test",
            "scored_documents": [_make_scored_doc()],
            "sub_queries": [],
        }
        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(
                {
                    "gaps_found": True,
                    "gaps": [
                        {"description": "gap", "query": "follow up query", "priority": 1},
                    ],
                },
                {
                    "agent": "gap_analyzer",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        result = await analyze_gaps(state, mock_caller)
        assert len(result["sub_queries"]) == 1
        assert result["sub_queries"] == result["follow_up_queries"]


class TestGapAnalysisGraphWiring:
    """Verify gap_analysis is wired into the graph."""

    def test_gap_analysis_node_in_graph(self):
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "gap_analysis" in source
        assert "search_followup" in source
        assert "extract_followup" in source
        assert "score_merge" in source

    def test_gap_analysis_conditional_edge(self):
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "should_follow_up" in source

    def test_follow_up_budget_tunable_exists(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        r = TunableRegistry()
        assert "follow_up_budget" in r
        assert r.get("follow_up_budget") == 5

    def test_state_has_follow_up_fields(self):
        """ResearchState should have follow_up_queries and follow_up_round."""
        import typing

        from deep_research_swarm.graph.state import ResearchState

        hints = typing.get_type_hints(ResearchState, include_extras=True)
        assert "follow_up_queries" in hints
        assert "follow_up_round" in hints

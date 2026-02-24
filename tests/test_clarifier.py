"""Tests for clarifier agent — pre-research scope analysis (V9)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deep_research_swarm.agents.clarifier import _infer_scope_hints, clarify


class TestInferScopeHints:
    def test_academic_question(self):
        hints = _infer_scope_hints("What do peer-reviewed studies say about CRISPR gene editing?")
        assert hints["domain"] == "academic"
        assert hints["depth"] == "comprehensive"

    def test_technical_question(self):
        hints = _infer_scope_hints("How do I implement a REST API with FastAPI?")
        assert hints["domain"] == "technical"

    def test_broad_question(self):
        hints = _infer_scope_hints("Give me a comprehensive overview of quantum computing")
        assert hints["breadth"] == "broad"
        assert hints["depth"] == "overview"

    def test_narrow_question(self):
        hints = _infer_scope_hints("What is the specific mechanism of mRNA vaccines?")
        assert hints["breadth"] == "narrow"

    def test_recent_question(self):
        hints = _infer_scope_hints("What are the latest developments in AI regulation 2026?")
        assert hints["recency"] == "recent"

    def test_historical_question(self):
        hints = _infer_scope_hints("What is the history of the internet?")
        assert hints["recency"] == "historical"

    def test_policy_question(self):
        hints = _infer_scope_hints("What government regulations apply to drone use?")
        assert hints["domain"] == "policy"

    def test_general_question(self):
        hints = _infer_scope_hints("What are the health benefits of meditation?")
        assert hints["domain"] == "general"
        assert hints["breadth"] == "moderate"
        assert hints["recency"] == "balanced"
        assert hints["depth"] == "detailed"

    def test_returns_all_keys(self):
        hints = _infer_scope_hints("anything")
        assert "breadth" in hints
        assert "depth" in hints
        assert "recency" in hints
        assert "domain" in hints


class TestClarifyAutoMode:
    @pytest.mark.asyncio
    async def test_auto_mode_no_llm_call(self):
        state = {"research_question": "What is quantum computing?"}
        result = await clarify(state, caller=None, mode="auto")
        assert "scope_hints" in result
        assert "token_usage" not in result

    @pytest.mark.asyncio
    async def test_auto_mode_returns_hints(self):
        state = {"research_question": "Latest peer-reviewed research on CRISPR"}
        result = await clarify(state, mode="auto")
        hints = result["scope_hints"]
        assert hints["domain"] == "academic"
        assert hints["recency"] == "recent"


class TestClarifyHitlMode:
    @pytest.mark.asyncio
    async def test_hitl_mode_calls_llm(self):
        state = {"research_question": "What is quantum computing?"}
        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(
                {
                    "scope_hints": {
                        "breadth": "broad",
                        "depth": "detailed",
                        "recency": "balanced",
                        "domain": "technical",
                    },
                    "clarifying_questions": [
                        "Are you interested in theoretical or applied quantum computing?"
                    ],
                },
                {
                    "agent": "clarifier",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        result = await clarify(state, mock_caller, mode="hitl")
        assert "scope_hints" in result
        assert result["scope_hints"]["breadth"] == "broad"
        assert "_clarifying_questions" in result
        assert len(result["_clarifying_questions"]) == 1


class TestClarifyGraphWiring:
    def test_clarify_node_in_graph(self):
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "clarify" in source

    def test_state_has_scope_hints(self):
        import typing

        from deep_research_swarm.graph.state import ResearchState

        hints = typing.get_type_hints(ResearchState, include_extras=True)
        assert "scope_hints" in hints

    def test_clarify_before_plan_in_graph(self):
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(anthropic_api_key="test-key")
        graph = build_graph(settings, enable_cache=False)
        edges = set()
        g = graph.get_graph()
        for edge in g.edges:
            edges.add((edge.source, edge.target))

        assert ("health_check", "clarify") in edges
        assert ("clarify", "plan") in edges
        # health_check should no longer go directly to plan
        assert ("health_check", "plan") not in edges

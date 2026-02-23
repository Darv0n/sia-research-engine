"""Tests for scoring/routing.py — backend routing (PR-07)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deep_research_swarm.scoring.routing import (
    QueryType,
    classify_query,
    route_backends,
)


class TestClassifyQueryAcademic:
    def test_research_paper_query(self):
        query = "peer-reviewed research on climate change effects"
        assert classify_query(query) == QueryType.ACADEMIC

    def test_doi_in_query(self):
        query = "paper about 10.1038/s41586-023-06768-0"
        assert classify_query(query) == QueryType.ACADEMIC

    def test_arxiv_id_in_query(self):
        query = "explain the findings of 2401.12345"
        assert classify_query(query) == QueryType.ACADEMIC

    def test_journal_study_query(self):
        query = "published study in journal about CRISPR methodology"
        assert classify_query(query) == QueryType.ACADEMIC

    def test_meta_analysis_query(self):
        query = "meta-analysis of systematic review on drug efficacy findings"
        assert classify_query(query) == QueryType.ACADEMIC

    def test_et_al_citation(self):
        query = "Smith et al. findings on quantum entanglement hypothesis"
        assert classify_query(query) == QueryType.ACADEMIC


class TestClassifyQueryArchival:
    def test_wayback_query(self):
        query = "wayback snapshot of the original version of this page"
        assert classify_query(query) == QueryType.ARCHIVAL

    def test_deleted_content(self):
        query = "deleted content that was previously available on that site"
        assert classify_query(query) == QueryType.ARCHIVAL

    def test_url_biases_archival(self):
        assert classify_query("https://example.com/removed-page") == QueryType.ARCHIVAL

    def test_historical_old_year(self):
        query = "historical web design trends from 1998"
        assert classify_query(query) == QueryType.ARCHIVAL


class TestClassifyQueryTechnical:
    def test_api_documentation(self):
        query = "documentation for the React framework API implementation"
        assert classify_query(query) == QueryType.TECHNICAL

    def test_github_library(self):
        query = "github library for implementing OAuth code tutorial"
        assert classify_query(query) == QueryType.TECHNICAL

    def test_pip_package(self):
        query = "pip package for error handling debug implementation"
        assert classify_query(query) == QueryType.TECHNICAL


class TestClassifyQueryGeneral:
    def test_no_signal_query(self):
        assert classify_query("best restaurants in Paris") == QueryType.GENERAL

    def test_short_query(self):
        assert classify_query("quantum computing") == QueryType.GENERAL

    def test_very_short_query(self):
        assert classify_query("hello world") == QueryType.GENERAL


class TestClassifyQueryMixed:
    def test_academic_beats_technical_tie(self):
        """ACADEMIC > TECHNICAL in tiebreaker priority."""
        query = "research paper on api implementation methodology"
        assert classify_query(query) == QueryType.ACADEMIC

    def test_dominant_type_wins(self):
        """Higher signal count wins regardless of tiebreaker."""
        query = "github npm pip documentation api library framework"
        assert classify_query(query) == QueryType.TECHNICAL


class TestRouteBackends:
    def test_academic_all_available(self):
        available = [
            "searxng",
            "exa",
            "tavily",
            "openalex",
            "semantic_scholar",
            "wayback",
        ]
        result = route_backends(QueryType.ACADEMIC, available)
        assert result == ["openalex", "semantic_scholar", "searxng"]

    def test_general_all_available(self):
        available = ["searxng", "exa", "tavily"]
        result = route_backends(QueryType.GENERAL, available)
        assert result == ["searxng", "exa", "tavily"]

    def test_archival_all_available(self):
        available = ["searxng", "wayback"]
        result = route_backends(QueryType.ARCHIVAL, available)
        assert result == ["wayback", "searxng"]

    def test_limited_backends(self):
        """Only returns backends that are actually available."""
        result = route_backends(QueryType.ACADEMIC, ["searxng"])
        assert result == ["searxng"]

    def test_no_preferred_available_falls_back(self):
        """Falls back to general chain when no preferred backends available."""
        result = route_backends(QueryType.ARCHIVAL, ["tavily"])
        assert result == ["tavily"]

    def test_empty_available_returns_searxng(self):
        """Last resort: returns searxng even if not available."""
        result = route_backends(QueryType.GENERAL, [])
        assert result == ["searxng"]

    def test_always_returns_at_least_one(self):
        for qt in QueryType:
            result = route_backends(qt, ["searxng"])
            assert len(result) >= 1

    def test_unknown_backend_not_returned(self):
        """Backends not in available list are filtered out."""
        result = route_backends(QueryType.ACADEMIC, ["searxng"])
        assert "openalex" not in result
        assert "semantic_scholar" not in result


class TestPlannerIntegration:
    @pytest.mark.asyncio
    async def test_user_backends_take_precedence(self):
        """User-specified backends override routing."""
        from deep_research_swarm.agents.planner import plan

        mock_data = {
            "perspectives": ["p1"],
            "sub_queries": [
                {
                    "question": "peer-reviewed research on climate change",
                    "perspective": "p1",
                    "priority": 1,
                }
            ],
        }
        mock_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.01,
        }

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(mock_data, mock_usage),
        )

        state = {
            "research_question": "climate change",
            "search_backends": ["exa", "tavily"],
            "current_iteration": 0,
        }

        result = await plan(
            state,
            mock_caller,
            available_backends=["searxng", "exa", "tavily"],
        )
        assert len(result["sub_queries"]) == 1
        assert result["sub_queries"][0]["search_backends"] == ["exa", "tavily"]

    @pytest.mark.asyncio
    async def test_routing_applied_when_no_user_backends(self):
        """Without user backends, routing classifies and routes."""
        from deep_research_swarm.agents.planner import plan

        mock_data = {
            "perspectives": ["p1"],
            "sub_queries": [
                {
                    "question": "peer-reviewed research on CRISPR methodology findings",
                    "perspective": "p1",
                    "priority": 1,
                }
            ],
        }
        mock_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.01,
        }

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(
            return_value=(mock_data, mock_usage),
        )

        state = {
            "research_question": "CRISPR",
            "search_backends": [],
            "current_iteration": 0,
        }

        available = ["searxng", "openalex", "semantic_scholar"]
        result = await plan(
            state,
            mock_caller,
            available_backends=available,
        )
        assert len(result["sub_queries"]) == 1
        sq_backends = result["sub_queries"][0]["search_backends"]
        assert "openalex" in sq_backends or "semantic_scholar" in sq_backends

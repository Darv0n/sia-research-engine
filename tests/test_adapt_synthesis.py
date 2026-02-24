"""Tests for adapt_synthesis overseer node."""

from __future__ import annotations

from deep_research_swarm.adaptive.adapt_synthesis import adapt_synthesis_node
from deep_research_swarm.adaptive.registry import TunableRegistry


class TestAdaptSynthesisNode:
    def test_returns_required_keys(self):
        result = adapt_synthesis_node({})
        assert "tunable_snapshot" in result
        assert "adaptation_events" in result
        assert "complexity_profile" in result

    def test_empty_state_valid_output(self):
        result = adapt_synthesis_node({})
        snap = result["tunable_snapshot"]
        assert isinstance(snap, dict)
        assert len(snap) > 0

    def test_high_volume_scales_citation_budget(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
            "scored_documents": [{"id": str(i)} for i in range(200)],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["citation_chain_budget"] > 50  # default

    def test_high_scored_docs_scales_contradiction_max(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(500)],
            "scored_documents": [{"id": str(i)} for i in range(100)],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["contradiction_max_docs"] > 10  # default

    def test_high_spend_rate_reduces_refinement(self):
        state = {
            "token_budget": 100000,
            "token_usage": [
                {"input_tokens": 40000, "output_tokens": 35000},
            ],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["max_refinement_attempts"] == 1

    def test_high_spend_rate_tightens_budget_exhaustion(self):
        state = {
            "token_budget": 100000,
            "token_usage": [
                {"input_tokens": 30000, "output_tokens": 25000},
            ],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["budget_exhaustion_pct"] == 0.8

    def test_events_have_correct_trigger(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
        }
        result = adapt_synthesis_node(state)
        for event in result["adaptation_events"]:
            assert event["trigger"] == "adapt_synthesis"

    def test_events_have_iteration(self):
        state = {
            "current_iteration": 3,
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(500)],
        }
        result = adapt_synthesis_node(state)
        for event in result["adaptation_events"]:
            assert event["iteration"] == 3

    def test_restores_from_snapshot(self):
        snap = TunableRegistry().snapshot()
        snap["citation_chain_budget"] = 100
        state = {
            "tunable_snapshot": snap,
            "search_results": [],
        }
        result = adapt_synthesis_node(state)
        # Should start from restored state
        assert isinstance(result["tunable_snapshot"], dict)

    def test_max_sections_scales_with_complexity(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["max_sections"] >= 7  # default or higher

    def test_max_docs_for_outline_scales(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["max_docs_for_outline"] >= 20  # default or higher

    def test_values_stay_within_bounds(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(100000)],
            "scored_documents": [{"id": str(i)} for i in range(500)],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert 20 <= snap["citation_chain_budget"] <= 150
        assert 5 <= snap["contradiction_max_docs"] <= 30
        assert 4 <= snap["max_sections"] <= 12
        assert 10 <= snap["max_docs_for_outline"] <= 50
        assert 1 <= snap["max_refinement_attempts"] <= 5

    def test_low_volume_does_not_increase(self):
        state = {
            "search_results": [{"id": "1", "backend": "searxng"}],
        }
        result = adapt_synthesis_node(state)
        snap = result["tunable_snapshot"]
        assert snap["citation_chain_budget"] <= 50
        assert snap["max_sections"] <= 7

    def test_complexity_profile_in_output(self):
        state = {
            "search_results": [{"id": "1", "backend": "exa"}],
            "scored_documents": [{"id": "1"}],
        }
        result = adapt_synthesis_node(state)
        profile = result["complexity_profile"]
        assert profile["scored_doc_count"] == 1
        assert 0.5 <= profile["multiplier"] <= 2.0

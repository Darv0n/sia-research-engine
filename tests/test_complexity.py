"""Tests for complexity analyzer — deterministic multiplier computation."""

from __future__ import annotations

from deep_research_swarm.adaptive.complexity import (
    compute_complexity_profile,
    compute_multiplier,
    compute_profile_from_state,
)

# --- compute_complexity_profile ---


class TestComputeComplexityProfile:
    def test_default_inputs_return_valid_profile(self):
        p = compute_complexity_profile()
        assert p["result_count"] == 0
        assert p["backends_used"] == 1
        assert p["iteration"] == 1
        assert 0.5 <= p["multiplier"] <= 2.0

    def test_average_case_near_one(self):
        """100 results, 2 backends, iteration 1 should yield multiplier ~1.0."""
        p = compute_complexity_profile(result_count=100, backends_used=2, iteration=1)
        assert 0.6 <= p["multiplier"] <= 1.1

    def test_high_volume_increases_multiplier(self):
        low = compute_complexity_profile(result_count=10, backends_used=1, iteration=1)
        high = compute_complexity_profile(result_count=3000, backends_used=1, iteration=1)
        assert high["multiplier"] > low["multiplier"]

    def test_more_backends_increases_multiplier(self):
        few = compute_complexity_profile(result_count=100, backends_used=1, iteration=1)
        many = compute_complexity_profile(result_count=100, backends_used=5, iteration=1)
        assert many["multiplier"] > few["multiplier"]

    def test_later_iterations_increase_multiplier(self):
        first = compute_complexity_profile(result_count=100, backends_used=2, iteration=1)
        third = compute_complexity_profile(result_count=100, backends_used=2, iteration=3)
        assert third["multiplier"] >= first["multiplier"]

    def test_multiplier_floor(self):
        """Minimum possible multiplier is 0.5."""
        p = compute_complexity_profile(result_count=1, backends_used=1, iteration=1)
        assert p["multiplier"] >= 0.5

    def test_multiplier_ceiling(self):
        """Maximum possible multiplier is 2.0."""
        p = compute_complexity_profile(result_count=100000, backends_used=10, iteration=10)
        assert p["multiplier"] <= 2.0

    def test_volume_factor_bounds(self):
        p = compute_complexity_profile(result_count=1)
        assert 0.5 <= p["volume_factor"] <= 2.0

    def test_backend_factor_floor(self):
        p = compute_complexity_profile(backends_used=0)
        assert p["backend_factor"] == 0.8

    def test_backend_factor_ceiling(self):
        p = compute_complexity_profile(backends_used=100)
        assert p["backend_factor"] == 1.5

    def test_iter_factor_floor(self):
        p = compute_complexity_profile(iteration=1)
        assert p["iter_factor"] == 1.0

    def test_iter_factor_ceiling(self):
        p = compute_complexity_profile(iteration=100)
        assert p["iter_factor"] == 1.3

    def test_passthrough_metrics(self):
        """Non-multiplier metrics are passed through unchanged."""
        p = compute_complexity_profile(
            extraction_success_rate=0.85,
            mean_grounding_score=0.72,
            token_spend_rate=0.3,
            scored_doc_count=120,
            citation_chain_yield=25,
        )
        assert p["extraction_success_rate"] == 0.85
        assert p["mean_grounding_score"] == 0.72
        assert p["token_spend_rate"] == 0.3
        assert p["scored_doc_count"] == 120
        assert p["citation_chain_yield"] == 25

    def test_values_are_rounded(self):
        p = compute_complexity_profile(result_count=123, backends_used=3, iteration=2)
        # All computed factors should have at most 4 decimal places
        for field in ("volume_factor", "backend_factor", "iter_factor", "multiplier"):
            val = p[field]
            assert val == round(val, 4)


# --- compute_multiplier ---


class TestComputeMultiplier:
    def test_extracts_multiplier(self):
        p = compute_complexity_profile(result_count=500, backends_used=3)
        assert compute_multiplier(p) == p["multiplier"]


# --- compute_profile_from_state ---


class TestComputeProfileFromState:
    def test_empty_state(self):
        p = compute_profile_from_state({})
        assert p["result_count"] == 0
        assert p["backends_used"] == 1
        assert 0.5 <= p["multiplier"] <= 2.0

    def test_with_search_results(self):
        state = {
            "search_results": [
                {"id": "1", "backend": "searxng"},
                {"id": "2", "backend": "searxng"},
                {"id": "3", "backend": "exa"},
            ],
        }
        p = compute_profile_from_state(state)
        assert p["result_count"] == 3
        assert p["backends_used"] == 2

    def test_extraction_success_rate(self):
        state = {
            "extracted_contents": [
                {"extraction_success": True},
                {"extraction_success": True},
                {"extraction_success": False},
                {"extraction_success": True},
            ],
        }
        p = compute_profile_from_state(state)
        assert p["extraction_success_rate"] == 0.75

    def test_mean_grounding_score(self):
        state = {
            "section_drafts": [
                {"grounding_score": 0.8},
                {"grounding_score": 0.6},
            ],
        }
        p = compute_profile_from_state(state)
        assert p["mean_grounding_score"] == 0.7

    def test_token_spend_rate(self):
        state = {
            "token_budget": 100000,
            "token_usage": [
                {"input_tokens": 10000, "output_tokens": 5000},
                {"input_tokens": 8000, "output_tokens": 2000},
            ],
        }
        p = compute_profile_from_state(state)
        assert p["token_spend_rate"] == 0.25  # 25000 / 100000

    def test_citation_chain_yield(self):
        state = {
            "citation_chain_results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
        }
        p = compute_profile_from_state(state)
        assert p["citation_chain_yield"] == 3

    def test_iteration_from_state(self):
        state = {"current_iteration": 3}
        p = compute_profile_from_state(state)
        assert p["iteration"] == 3

    def test_scored_doc_count(self):
        state = {
            "scored_documents": [{"id": "1"}, {"id": "2"}],
        }
        p = compute_profile_from_state(state)
        assert p["scored_doc_count"] == 2


# --- Specific scenarios from live test pain points ---


class TestLiveScenarios:
    def test_bobbin_lace_scenario(self):
        """2800 results, 3 backends, iteration 1 — should scale up."""
        p = compute_complexity_profile(result_count=2800, backends_used=3, iteration=1)
        assert p["multiplier"] > 1.2, "High-volume topic should get multiplier > 1.2"
        assert p["volume_factor"] > 1.5

    def test_simple_general_topic(self):
        """50 results, 1 backend — should scale down or stay neutral."""
        p = compute_complexity_profile(result_count=50, backends_used=1, iteration=1)
        assert p["multiplier"] <= 1.0

    def test_deep_academic_topic(self):
        """500 results, 4 backends, iteration 3 — max complexity."""
        p = compute_complexity_profile(result_count=500, backends_used=4, iteration=3)
        assert p["multiplier"] > 1.0

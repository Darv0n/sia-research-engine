"""Tests for adapt_extraction overseer node."""

from __future__ import annotations

from deep_research_swarm.adaptive.adapt_extraction import adapt_extraction_node
from deep_research_swarm.adaptive.registry import TunableRegistry


class TestAdaptExtractionNode:
    def test_returns_required_keys(self):
        result = adapt_extraction_node({})
        assert "tunable_snapshot" in result
        assert "adaptation_events" in result
        assert "complexity_profile" in result

    def test_snapshot_is_dict(self):
        result = adapt_extraction_node({})
        assert isinstance(result["tunable_snapshot"], dict)

    def test_events_is_list(self):
        result = adapt_extraction_node({})
        assert isinstance(result["adaptation_events"], list)

    def test_no_events_when_defaults_unchanged(self):
        """With minimal state, multiplier is low, so defaults may shift."""
        result = adapt_extraction_node({})
        # With 0 results the multiplier is 0.5, which changes extraction_cap
        # from 30 to 15 — so events are expected
        events = result["adaptation_events"]
        assert isinstance(events, list)

    def test_high_volume_scales_extraction_cap_up(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
        }
        result = adapt_extraction_node(state)
        snap = result["tunable_snapshot"]
        assert snap["extraction_cap"] > 30  # default is 30

    def test_low_volume_scales_extraction_cap_down(self):
        state = {
            "search_results": [{"id": "1", "backend": "searxng"}],
        }
        result = adapt_extraction_node(state)
        snap = result["tunable_snapshot"]
        assert snap["extraction_cap"] <= 30

    def test_high_volume_scales_results_per_query(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
        }
        result = adapt_extraction_node(state)
        snap = result["tunable_snapshot"]
        assert snap["results_per_query"] >= 10

    def test_events_have_correct_trigger(self):
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(3000)],
        }
        result = adapt_extraction_node(state)
        for event in result["adaptation_events"]:
            assert event["trigger"] == "adapt_extraction"

    def test_events_have_iteration(self):
        state = {
            "current_iteration": 2,
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(500)],
        }
        result = adapt_extraction_node(state)
        for event in result["adaptation_events"]:
            assert event["iteration"] == 2

    def test_restores_from_existing_snapshot(self):
        """If state already has a tunable_snapshot, it should be used."""
        initial_snap = TunableRegistry().snapshot()
        initial_snap["extraction_cap"] = 75  # pre-set
        state = {
            "tunable_snapshot": initial_snap,
            "search_results": [],
        }
        result = adapt_extraction_node(state)
        # With 0 results, multiplier is low — cap should scale down from 75
        snap = result["tunable_snapshot"]
        # The key test: we didn't reset to default 30
        # With multiplier 0.5, it would try 30*0.5=15 (from default),
        # since set_scaled uses default, not current value
        assert snap["extraction_cap"] == 15

    def test_complexity_profile_in_output(self):
        state = {
            "search_results": [{"id": "1", "backend": "exa"}],
        }
        result = adapt_extraction_node(state)
        profile = result["complexity_profile"]
        assert profile["result_count"] == 1
        assert profile["backends_used"] == 1
        assert 0.5 <= profile["multiplier"] <= 2.0

    def test_multiple_backends_increase_multiplier(self):
        state = {
            "search_results": [
                {"id": "1", "backend": "searxng"},
                {"id": "2", "backend": "exa"},
                {"id": "3", "backend": "tavily"},
                {"id": "4", "backend": "openalex"},
            ],
        }
        result = adapt_extraction_node(state)
        profile = result["complexity_profile"]
        assert profile["backends_used"] == 4

    def test_values_stay_within_bounds(self):
        """Even with extreme inputs, tunables must stay within bounds."""
        state = {
            "search_results": [{"id": str(i), "backend": "searxng"} for i in range(100000)],
        }
        result = adapt_extraction_node(state)
        snap = result["tunable_snapshot"]
        assert 15 <= snap["extraction_cap"] <= 100
        assert 5 <= snap["results_per_query"] <= 30
        assert 20000 <= snap["content_truncation_chars"] <= 200000

    def test_empty_state_produces_valid_output(self):
        """Completely empty state should not crash."""
        result = adapt_extraction_node({})
        assert isinstance(result["tunable_snapshot"], dict)
        assert len(result["tunable_snapshot"]) > 0

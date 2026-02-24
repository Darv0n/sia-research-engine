"""Tests for TunableRegistry — bounded parameter store for adaptive control."""

from __future__ import annotations

import pytest

from deep_research_swarm.adaptive.registry import TunableRegistry, _clamp
from deep_research_swarm.contracts import Tunable

# --- _clamp ---


class TestClamp:
    def test_clamp_within_range(self):
        assert _clamp(5, 1, 10) == 5

    def test_clamp_below_floor(self):
        assert _clamp(0, 1, 10) == 1

    def test_clamp_above_ceiling(self):
        assert _clamp(15, 1, 10) == 10

    def test_clamp_at_floor(self):
        assert _clamp(1, 1, 10) == 1

    def test_clamp_at_ceiling(self):
        assert _clamp(10, 1, 10) == 10

    def test_clamp_preserves_int_type(self):
        result = _clamp(5, 1, 10)
        assert isinstance(result, int)

    def test_clamp_preserves_float_type(self):
        result = _clamp(0.5, 0.1, 1.0)
        assert isinstance(result, float)

    def test_clamp_negative_values(self):
        assert _clamp(-5, -10, -1) == -5
        assert _clamp(-15, -10, -1) == -10


# --- TunableRegistry construction ---


class TestRegistryConstruction:
    def test_default_registry_has_all_tunables(self):
        r = TunableRegistry()
        expected = [
            "extraction_cap",
            "content_truncation_chars",
            "contradiction_max_docs",
            "budget_exhaustion_pct",
            "jaccard_threshold",
            "grounding_pass_threshold",
            "max_refinement_attempts",
            "max_passages_per_section",
            "citation_chain_budget",
            "citation_chain_max_hops",
            "citation_chain_top_seeds",
            "results_per_query",
            "min_sections",
            "max_sections",
            "max_docs_for_outline",
        ]
        for name in expected:
            assert name in r, f"{name} not in registry"

    def test_default_values_match_v7_hardcodes(self):
        r = TunableRegistry()
        assert r.get("extraction_cap") == 30
        assert r.get("content_truncation_chars") == 50000
        assert r.get("contradiction_max_docs") == 10
        assert r.get("jaccard_threshold") == 0.3
        assert r.get("grounding_pass_threshold") == 0.8
        assert r.get("max_refinement_attempts") == 2
        assert r.get("max_passages_per_section") == 8
        assert r.get("citation_chain_budget") == 50
        assert r.get("citation_chain_max_hops") == 2
        assert r.get("citation_chain_top_seeds") == 5
        assert r.get("results_per_query") == 10
        assert r.get("budget_exhaustion_pct") == 0.9
        assert r.get("min_sections") == 3
        assert r.get("max_sections") == 7
        assert r.get("max_docs_for_outline") == 20

    def test_len(self):
        r = TunableRegistry()
        assert len(r) == 15

    def test_contains(self):
        r = TunableRegistry()
        assert "extraction_cap" in r
        assert "nonexistent" not in r


# --- get/set ---


class TestGetSet:
    def test_get_known_tunable(self):
        r = TunableRegistry()
        assert r.get("extraction_cap") == 30

    def test_get_unknown_raises_keyerror(self):
        r = TunableRegistry()
        with pytest.raises(KeyError):
            r.get("totally_unknown")

    def test_set_within_bounds(self):
        r = TunableRegistry()
        result = r.set("extraction_cap", 50)
        assert result == 50
        assert r.get("extraction_cap") == 50

    def test_set_clamps_below_floor(self):
        r = TunableRegistry()
        result = r.set("extraction_cap", 5)
        assert result == 15  # floor
        assert r.get("extraction_cap") == 15

    def test_set_clamps_above_ceiling(self):
        r = TunableRegistry()
        result = r.set("extraction_cap", 500)
        assert result == 100  # ceiling
        assert r.get("extraction_cap") == 100

    def test_set_unknown_raises_keyerror(self):
        r = TunableRegistry()
        with pytest.raises(KeyError):
            r.set("nonexistent", 42)

    def test_set_float_tunable(self):
        r = TunableRegistry()
        result = r.set("jaccard_threshold", 0.25)
        assert result == 0.25
        assert r.get("jaccard_threshold") == 0.25

    def test_set_float_clamps_floor(self):
        r = TunableRegistry()
        result = r.set("jaccard_threshold", 0.05)
        assert result == 0.15  # floor

    def test_set_float_clamps_ceiling(self):
        r = TunableRegistry()
        result = r.set("jaccard_threshold", 0.9)
        assert result == 0.5  # ceiling


# --- set_scaled ---


class TestSetScaled:
    def test_scale_up(self):
        r = TunableRegistry()
        result = r.set_scaled("extraction_cap", 2.0)
        assert result == 60  # 30 * 2.0

    def test_scale_down(self):
        r = TunableRegistry()
        result = r.set_scaled("extraction_cap", 0.5)
        assert result == 15  # 30 * 0.5 = 15 (at floor)

    def test_scale_clamps_ceiling(self):
        r = TunableRegistry()
        result = r.set_scaled("extraction_cap", 5.0)
        assert result == 100  # 30 * 5.0 = 150, clamped to 100

    def test_scale_preserves_int(self):
        r = TunableRegistry()
        result = r.set_scaled("extraction_cap", 1.5)
        assert isinstance(result, int)
        assert result == 45  # round(30 * 1.5)

    def test_scale_float_tunable(self):
        r = TunableRegistry()
        result = r.set_scaled("jaccard_threshold", 0.8)
        assert result == 0.24  # 0.3 * 0.8 = 0.24


# --- get_default / get_definition ---


class TestMetadata:
    def test_get_default(self):
        r = TunableRegistry()
        assert r.get_default("extraction_cap") == 30

    def test_get_default_unchanged_by_set(self):
        r = TunableRegistry()
        r.set("extraction_cap", 75)
        assert r.get_default("extraction_cap") == 30
        assert r.get("extraction_cap") == 75

    def test_get_definition(self):
        r = TunableRegistry()
        defn = r.get_definition("extraction_cap")
        assert defn["name"] == "extraction_cap"
        assert defn["default"] == 30
        assert defn["floor"] == 15
        assert defn["ceiling"] == 100
        assert defn["category"] == "extraction"


# --- snapshot / from_snapshot ---


class TestSnapshot:
    def test_snapshot_roundtrip(self):
        r = TunableRegistry()
        r.set("extraction_cap", 75)
        r.set("jaccard_threshold", 0.2)

        snap = r.snapshot()
        r2 = TunableRegistry.from_snapshot(snap)

        assert r2.get("extraction_cap") == 75
        assert r2.get("jaccard_threshold") == 0.2

    def test_snapshot_contains_all_tunables(self):
        r = TunableRegistry()
        snap = r.snapshot()
        assert len(snap) == len(r)

    def test_from_snapshot_ignores_unknown_keys(self):
        r = TunableRegistry()
        snap = r.snapshot()
        snap["future_tunable"] = 999
        r2 = TunableRegistry.from_snapshot(snap)
        assert "future_tunable" not in r2

    def test_from_snapshot_partial(self):
        """A snapshot missing some tunables uses defaults for missing ones."""
        r2 = TunableRegistry.from_snapshot({"extraction_cap": 50})
        assert r2.get("extraction_cap") == 50
        assert r2.get("jaccard_threshold") == 0.3  # default

    def test_from_snapshot_clamps_values(self):
        """Values in snapshot beyond bounds get clamped."""
        r2 = TunableRegistry.from_snapshot({"extraction_cap": 999})
        assert r2.get("extraction_cap") == 100  # ceiling


# --- categories / names ---


class TestCategoriesAndNames:
    def test_names_returns_all(self):
        r = TunableRegistry()
        assert len(r.names()) == 15

    def test_categories_groups_correctly(self):
        r = TunableRegistry()
        cats = r.categories()
        assert "extraction" in cats
        assert "scoring" in cats
        assert "grounding" in cats
        assert "search" in cats
        assert "synthesis" in cats
        assert "extraction_cap" in cats["extraction"]

    def test_all_tunables_in_some_category(self):
        r = TunableRegistry()
        cats = r.categories()
        all_categorized = []
        for names in cats.values():
            all_categorized.extend(names)
        assert set(all_categorized) == set(r.names())


# --- diff_from_defaults ---


class TestDiffFromDefaults:
    def test_no_diff_on_fresh_registry(self):
        r = TunableRegistry()
        assert r.diff_from_defaults() == {}

    def test_diff_after_set(self):
        r = TunableRegistry()
        r.set("extraction_cap", 75)
        diff = r.diff_from_defaults()
        assert "extraction_cap" in diff
        assert diff["extraction_cap"]["default"] == 30
        assert diff["extraction_cap"]["current"] == 75
        assert diff["extraction_cap"]["category"] == "extraction"

    def test_diff_only_changed(self):
        r = TunableRegistry()
        r.set("extraction_cap", 75)
        diff = r.diff_from_defaults()
        assert len(diff) == 1


# --- custom registration ---


class TestCustomRegistration:
    def test_register_custom_tunable(self):
        r = TunableRegistry()
        t = Tunable(name="custom_param", default=42, floor=10, ceiling=100, category="custom")
        r.register(t)
        assert r.get("custom_param") == 42
        assert "custom_param" in r

    def test_register_overwrites_existing(self):
        r = TunableRegistry()
        t = Tunable(name="extraction_cap", default=50, floor=10, ceiling=200, category="extraction")
        r.register(t)
        assert r.get("extraction_cap") == 50
        assert r.get_definition("extraction_cap")["ceiling"] == 200


# --- Invariant I5: safety floors ---


class TestSafetyInvariant:
    """Verify that I5 (adaptive, not chaotic) holds: floors prevent dangerous values."""

    def test_grounding_cannot_be_disabled(self):
        """jaccard_threshold has floor 0.15 — can't go to 0."""
        r = TunableRegistry()
        result = r.set("jaccard_threshold", 0.0)
        assert result == 0.15

    def test_extraction_cap_cannot_be_zero(self):
        r = TunableRegistry()
        result = r.set("extraction_cap", 0)
        assert result == 15

    def test_refinement_always_at_least_one(self):
        r = TunableRegistry()
        result = r.set("max_refinement_attempts", 0)
        assert result == 1

    def test_budget_exhaustion_floor(self):
        r = TunableRegistry()
        result = r.set("budget_exhaustion_pct", 0.1)
        assert result == 0.7

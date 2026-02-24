"""Tests for reporting/adaptive_section.py — adaptive adjustments report (PR-11)."""

from __future__ import annotations

from deep_research_swarm.contracts import AdaptationEvent, ComplexityProfile
from deep_research_swarm.reporting.adaptive_section import render_adaptive_section


def _make_event(
    name: str = "extraction_cap",
    old: int | float = 30,
    new: int | float = 60,
    reason: str = "high_volume",
    trigger: str = "adapt_extraction",
    iteration: int = 1,
) -> AdaptationEvent:
    return AdaptationEvent(
        tunable_name=name,
        old_value=old,
        new_value=new,
        reason=reason,
        trigger=trigger,
        iteration=iteration,
    )


def _make_profile(**kwargs) -> ComplexityProfile:
    defaults = {
        "result_count": 200,
        "backends_used": 3,
        "iteration": 1,
        "extraction_success_rate": 0.85,
        "mean_grounding_score": 0.7,
        "token_spend_rate": 0.3,
        "scored_doc_count": 50,
        "citation_chain_yield": 25,
        "volume_factor": 1.15,
        "backend_factor": 1.1,
        "iter_factor": 1.0,
        "multiplier": 1.3,
    }
    defaults.update(kwargs)
    return ComplexityProfile(**defaults)


class TestRenderAdaptiveSection:
    def test_empty_returns_empty(self):
        assert render_adaptive_section([], None) == ""

    def test_events_only(self):
        events = [_make_event()]
        result = render_adaptive_section(events)
        assert "## Adaptive Adjustments" in result
        assert "extraction_cap" in result
        assert "30" in result
        assert "60" in result

    def test_profile_only(self):
        profile = _make_profile()
        result = render_adaptive_section([], profile)
        assert "## Adaptive Adjustments" in result
        assert "1.30" in result  # multiplier
        assert "200 results" in result
        assert "3 backends" in result

    def test_both(self):
        events = [_make_event(), _make_event(name="jaccard_threshold", old=0.3, new=0.2)]
        profile = _make_profile()
        result = render_adaptive_section(events, profile)
        assert "## Adaptive Adjustments" in result
        assert "Complexity multiplier" in result
        assert "extraction_cap" in result
        assert "jaccard_threshold" in result

    def test_table_headers(self):
        events = [_make_event()]
        result = render_adaptive_section(events)
        assert "| Tunable |" in result
        assert "| Old |" in result
        assert "| New |" in result
        assert "| Trigger |" in result
        assert "| Reason |" in result

    def test_float_formatting(self):
        events = [_make_event(name="jaccard_threshold", old=0.300, new=0.200)]
        result = render_adaptive_section(events)
        assert "0.300" in result
        assert "0.200" in result

    def test_trigger_shown(self):
        events = [_make_event(trigger="adapt_synthesis")]
        result = render_adaptive_section(events)
        assert "adapt_synthesis" in result

    def test_reason_shown(self):
        events = [_make_event(reason="token_pacing")]
        result = render_adaptive_section(events)
        assert "token_pacing" in result

    def test_multiple_events(self):
        events = [
            _make_event(name="extraction_cap", old=30, new=60),
            _make_event(name="results_per_query", old=10, new=20),
            _make_event(name="citation_chain_budget", old=50, new=100),
        ]
        result = render_adaptive_section(events)
        lines = [ln for ln in result.split("\n") if ln.startswith("|") and "---" not in ln]
        # Header + 3 data rows
        assert len(lines) == 4

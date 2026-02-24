"""Adapt-extraction overseer node — runs between search and extract.

Observes search result volume and adjusts extraction-phase tunables.
Deterministic, no LLM, sub-millisecond.
"""

from __future__ import annotations

from deep_research_swarm.adaptive.complexity import compute_profile_from_state
from deep_research_swarm.adaptive.registry import TunableRegistry
from deep_research_swarm.contracts import AdaptationEvent


def adapt_extraction_node(state: dict) -> dict:
    """Overseer node: observe search results, adjust extraction tunables.

    Placed between search -> extract in the graph.
    Returns updated tunable_snapshot, adaptation_events, and complexity_profile.
    """
    # Restore or create registry
    snap = state.get("tunable_snapshot", {})
    registry = TunableRegistry.from_snapshot(snap) if snap else TunableRegistry()

    # Compute complexity from current state
    profile = compute_profile_from_state(state)
    multiplier = profile["multiplier"]
    iteration = state.get("current_iteration", 1)

    events: list[AdaptationEvent] = []

    # --- Adjust extraction_cap based on result volume ---
    old_cap = registry.get("extraction_cap")
    new_cap = registry.set_scaled("extraction_cap", multiplier)
    if new_cap != old_cap:
        events.append(
            AdaptationEvent(
                tunable_name="extraction_cap",
                old_value=old_cap,
                new_value=new_cap,
                reason=f"Result volume {profile['result_count']}, multiplier {multiplier:.2f}",
                trigger="adapt_extraction",
                iteration=iteration,
            )
        )

    # --- Adjust results_per_query based on backend diversity ---
    old_rpq = registry.get("results_per_query")
    new_rpq = registry.set_scaled("results_per_query", multiplier)
    if new_rpq != old_rpq:
        events.append(
            AdaptationEvent(
                tunable_name="results_per_query",
                old_value=old_rpq,
                new_value=new_rpq,
                reason=f"Backend diversity {profile['backends_used']}, multiplier {multiplier:.2f}",
                trigger="adapt_extraction",
                iteration=iteration,
            )
        )

    # --- Adjust content_truncation_chars proportionally ---
    old_trunc = registry.get("content_truncation_chars")
    new_trunc = registry.set_scaled("content_truncation_chars", multiplier)
    if new_trunc != old_trunc:
        events.append(
            AdaptationEvent(
                tunable_name="content_truncation_chars",
                old_value=old_trunc,
                new_value=new_trunc,
                reason=f"Scaled with complexity multiplier {multiplier:.2f}",
                trigger="adapt_extraction",
                iteration=iteration,
            )
        )

    return {
        "tunable_snapshot": registry.snapshot(),
        "adaptation_events": events,
        "complexity_profile": dict(profile),
    }

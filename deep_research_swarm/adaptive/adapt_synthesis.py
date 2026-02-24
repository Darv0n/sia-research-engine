"""Adapt-synthesis overseer node — runs between score and citation_chain.

Observes scored document count, grounding scores, and token spend rate.
Adjusts synthesis-phase tunables (section counts, grounding thresholds,
citation chain budget). Deterministic, no LLM.
"""

from __future__ import annotations

from deep_research_swarm.adaptive.complexity import compute_profile_from_state
from deep_research_swarm.adaptive.registry import TunableRegistry
from deep_research_swarm.contracts import AdaptationEvent


def adapt_synthesis_node(state: dict) -> dict:
    """Overseer node: observe scoring output, adjust synthesis tunables.

    Placed between score -> citation_chain in the graph.
    """
    snap = state.get("tunable_snapshot", {})
    registry = TunableRegistry.from_snapshot(snap) if snap else TunableRegistry()

    profile = compute_profile_from_state(state)
    multiplier = profile["multiplier"]
    iteration = state.get("current_iteration", 1)

    events: list[AdaptationEvent] = []

    # --- Scale citation chain budget with complexity ---
    old_budget = registry.get("citation_chain_budget")
    new_budget = registry.set_scaled("citation_chain_budget", multiplier)
    if new_budget != old_budget:
        events.append(
            AdaptationEvent(
                tunable_name="citation_chain_budget",
                old_value=old_budget,
                new_value=new_budget,
                reason=f"Complexity multiplier {multiplier:.2f}",
                trigger="adapt_synthesis",
                iteration=iteration,
            )
        )

    # --- Scale contradiction_max_docs with scored doc count ---
    scored_count = profile["scored_doc_count"]
    old_contra = registry.get("contradiction_max_docs")
    # Scale: if many scored docs, allow contradiction detector to see more
    if scored_count > 50:
        contra_scale = min(multiplier * 1.5, 3.0)
    else:
        contra_scale = multiplier
    new_contra = registry.set_scaled("contradiction_max_docs", contra_scale)
    if new_contra != old_contra:
        events.append(
            AdaptationEvent(
                tunable_name="contradiction_max_docs",
                old_value=old_contra,
                new_value=new_contra,
                reason=f"Scored docs {scored_count}, scale {contra_scale:.2f}",
                trigger="adapt_synthesis",
                iteration=iteration,
            )
        )

    # --- Adjust section counts with complexity ---
    old_max_sec = registry.get("max_sections")
    new_max_sec = registry.set_scaled("max_sections", multiplier)
    if new_max_sec != old_max_sec:
        events.append(
            AdaptationEvent(
                tunable_name="max_sections",
                old_value=old_max_sec,
                new_value=new_max_sec,
                reason=f"Complexity multiplier {multiplier:.2f}",
                trigger="adapt_synthesis",
                iteration=iteration,
            )
        )

    old_docs_outline = registry.get("max_docs_for_outline")
    new_docs_outline = registry.set_scaled("max_docs_for_outline", multiplier)
    if new_docs_outline != old_docs_outline:
        events.append(
            AdaptationEvent(
                tunable_name="max_docs_for_outline",
                old_value=old_docs_outline,
                new_value=new_docs_outline,
                reason=f"Complexity multiplier {multiplier:.2f}",
                trigger="adapt_synthesis",
                iteration=iteration,
            )
        )

    # --- Adjust grounding thresholds based on token spend rate ---
    spend_rate = profile["token_spend_rate"]
    if spend_rate > 0.7:
        # Running low on budget — reduce refinement attempts to save tokens
        old_refine = registry.get("max_refinement_attempts")
        new_refine = registry.set("max_refinement_attempts", 1)
        if new_refine != old_refine:
            events.append(
                AdaptationEvent(
                    tunable_name="max_refinement_attempts",
                    old_value=old_refine,
                    new_value=new_refine,
                    reason=f"Token spend rate {spend_rate:.2f} > 0.7, conserving budget",
                    trigger="adapt_synthesis",
                    iteration=iteration,
                )
            )
    elif multiplier > 1.3:
        # Complex topic — allow more refinement
        old_refine = registry.get("max_refinement_attempts")
        new_refine = registry.set_scaled("max_refinement_attempts", multiplier)
        if new_refine != old_refine:
            events.append(
                AdaptationEvent(
                    tunable_name="max_refinement_attempts",
                    old_value=old_refine,
                    new_value=new_refine,
                    reason=f"High complexity (multiplier {multiplier:.2f}), more refinement",
                    trigger="adapt_synthesis",
                    iteration=iteration,
                )
            )

    # --- Adjust budget exhaustion threshold based on spend rate ---
    if spend_rate > 0.5:
        # Past halfway — lower the exhaustion threshold to trigger earlier convergence
        old_exhaust = registry.get("budget_exhaustion_pct")
        new_exhaust = registry.set("budget_exhaustion_pct", 0.8)
        if new_exhaust != old_exhaust:
            events.append(
                AdaptationEvent(
                    tunable_name="budget_exhaustion_pct",
                    old_value=old_exhaust,
                    new_value=new_exhaust,
                    reason=f"Token spend rate {spend_rate:.2f} > 0.5, tighter budget gate",
                    trigger="adapt_synthesis",
                    iteration=iteration,
                )
            )

    return {
        "tunable_snapshot": registry.snapshot(),
        "adaptation_events": events,
        "complexity_profile": dict(profile),
    }

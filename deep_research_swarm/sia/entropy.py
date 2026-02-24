"""Thermodynamic entropy layer for convergence control.

Pure deterministic computation from ResearchState observables. Zero LLM calls.

Entropy e in [0, 1] measures unresolved variance across research dimensions:
  e_amb:   ambiguity — MEDIUM confidence sections + unresolved gaps
  e_conf:  conflict — contradiction density + grader score divergence
  e_nov:   novelty — new query ratio + outline heading delta
  e_trust: trust/coherence — grounding score variance + unmapped citations

Scalar: e = 0.30*amb + 0.30*conf + 0.20*nov + 0.20*trust

Four bands with distinct steering policies:
  crystalline (<=0.20): synthesis zone — do not introduce catalysts
  convergence (0.20-0.45): build and compress — prioritize builders
  turbulence  (0.45-0.70): productive tension — balanced collision
  runaway     (>0.70): containment needed — stabilize immediately
"""

from __future__ import annotations

import statistics
from typing import Any

from deep_research_swarm.contracts import EntropyBand, EntropyState


def classify_band(e: float) -> str:
    """Classify entropy value into a thermodynamic band."""
    if e <= 0.20:
        return EntropyBand.CRYSTALLINE.value
    if e <= 0.45:
        return EntropyBand.CONVERGENCE.value
    if e <= 0.70:
        return EntropyBand.TURBULENCE.value
    return EntropyBand.RUNAWAY.value


def compute_entropy(
    state: dict[str, Any],
    prev_entropy: EntropyState | None = None,
) -> EntropyState:
    """Compute entropy from ResearchState observables.

    All signals are normalized to [0, 1]. Higher = more unresolved.
    """
    e_amb = _compute_ambiguity(state)
    e_conf = _compute_conflict(state)
    e_nov = _compute_novelty(state, prev_entropy)
    e_trust = _compute_trust(state)

    e = 0.30 * e_amb + 0.30 * e_conf + 0.20 * e_nov + 0.20 * e_trust
    e = max(0.0, min(1.0, e))

    prev_turn = prev_entropy["turn"] if prev_entropy else 0
    prev_stag = prev_entropy["stagnation_count"] if prev_entropy else 0

    # Stagnation: entropy delta < 0.03
    if prev_entropy and abs(e - prev_entropy["e"]) < 0.03:
        stagnation_count = prev_stag + 1
    else:
        stagnation_count = 0

    return EntropyState(
        e=round(e, 4),
        e_amb=round(e_amb, 4),
        e_conf=round(e_conf, 4),
        e_nov=round(e_nov, 4),
        e_trust=round(e_trust, 4),
        band=classify_band(e),
        turn=prev_turn + 1,
        stagnation_count=stagnation_count,
    )


def entropy_gate(
    entropy: EntropyState,
    section_drafts: list[dict],
) -> tuple[bool, str]:
    """Check if synthesis is allowed given current entropy.

    Returns (allowed, reason).
    Blocks synthesis if entropy is in runaway band or turbulence without sections.
    """
    if entropy["band"] == EntropyBand.RUNAWAY.value:
        return False, f"entropy_runaway (e={entropy['e']:.3f})"

    if entropy["band"] == EntropyBand.TURBULENCE.value and not section_drafts:
        return False, f"turbulence_no_sections (e={entropy['e']:.3f})"

    return True, f"entropy_ok (e={entropy['e']:.3f}, band={entropy['band']})"


def detect_false_convergence(
    entropy: EntropyState,
    section_drafts: list[dict],
    contradictions: list[dict],
    prev_entropy: EntropyState | None = None,
) -> tuple[bool, str]:
    """Detect false convergence — low entropy without genuine constraint resolution.

    False convergence signals:
    1. Low entropy but unresolved contradictions present
    2. Rapid entropy drop without constraint gain
    3. High agreement (all sections HIGH) without diverse sources
    """
    # Signal 1: Low entropy + unresolved contradictions
    if entropy["e"] <= 0.30 and len(contradictions) >= 2 and entropy["e_conf"] > 0.3:
        return True, (
            f"low_entropy_with_contradictions "
            f"(e={entropy['e']:.3f}, contradictions={len(contradictions)})"
        )

    # Signal 2: Rapid drop without constraint gain
    if prev_entropy is not None:
        delta = prev_entropy["e"] - entropy["e"]
        if delta > 0.25:
            # Big drop — check if sections actually improved
            if section_drafts:
                scores = [s.get("confidence_score", 0) for s in section_drafts]
                avg = sum(scores) / len(scores)
                if avg < 0.7:
                    return True, (
                        f"rapid_drop_without_quality (delta={delta:.3f}, avg_confidence={avg:.3f})"
                    )

    # Signal 3: Uniform high scores (suspiciously perfect)
    if section_drafts and len(section_drafts) >= 3:
        scores = [s.get("confidence_score", 0) for s in section_drafts]
        if all(s >= 0.9 for s in scores) and len(contradictions) > 0:
            return True, (
                f"suspicious_uniformity (all_scores>=0.9, contradictions={len(contradictions)})"
            )

    return False, "no_false_convergence"


def detect_dominance(
    entropy_history: list[EntropyState],
    section_drafts: list[dict],
) -> tuple[bool, str]:
    """Detect single-perspective dominance.

    Dominance signals:
    1. All grader scores converge to same value (no dimension spread)
    2. All section confidence identical (cookie-cutter synthesis)
    """
    if not section_drafts or len(section_drafts) < 2:
        return False, "insufficient_sections"

    # Check grader dimension spread
    scores = [s.get("confidence_score", 0.5) for s in section_drafts]
    try:
        stdev = statistics.stdev(scores)
    except statistics.StatisticsError:
        stdev = 0.0

    if stdev < 0.01 and len(section_drafts) >= 3:
        return True, (f"uniform_confidence (stdev={stdev:.4f}, sections={len(section_drafts)})")

    # Check grader dimension convergence (all dimensions scoring the same)
    grader_spreads = []
    for sec in section_drafts:
        gs = sec.get("grader_scores", {})
        if gs:
            dims = [
                gs.get("relevance", 0.5),
                gs.get("hallucination", 0.5),
                gs.get("quality", 0.5),
            ]
            try:
                grader_spreads.append(statistics.stdev(dims))
            except statistics.StatisticsError:
                pass

    if grader_spreads and all(s < 0.02 for s in grader_spreads):
        return True, "grader_dimension_collapse (all dimensions identical)"

    return False, "no_dominance"


# ============================================================
# Internal Signal Extractors
# ============================================================


def _compute_ambiguity(state: dict[str, Any]) -> float:
    """e_amb: MEDIUM confidence sections + unresolved gaps."""
    sections = state.get("section_drafts", [])
    gaps = state.get("research_gaps", [])

    if not sections:
        return 0.85  # high ambiguity with no sections

    # Ratio of non-HIGH sections
    non_high = sum(1 for s in sections if s.get("confidence_score", 0) < 0.8)
    section_ambiguity = non_high / len(sections) if sections else 0.5

    # Gap pressure
    gap_pressure = min(1.0, len(gaps) / 5.0)  # normalize: 5+ gaps = max

    return 0.6 * section_ambiguity + 0.4 * gap_pressure


def _compute_conflict(state: dict[str, Any]) -> float:
    """e_conf: contradiction density + grader score divergence."""
    contradictions = state.get("contradictions", [])
    sections = state.get("section_drafts", [])
    scored_docs = state.get("scored_documents", [])

    # Contradiction density
    doc_count = max(len(scored_docs), 1)
    contradiction_density = min(1.0, len(contradictions) / max(doc_count * 0.2, 1))

    # Grader score divergence (spread across dimensions)
    divergences = []
    for sec in sections:
        gs = sec.get("grader_scores", {})
        if gs:
            dims = [
                gs.get("relevance", 0.5),
                gs.get("hallucination", 0.5),
                gs.get("quality", 0.5),
            ]
            try:
                divergences.append(statistics.stdev(dims))
            except statistics.StatisticsError:
                pass

    avg_divergence = sum(divergences) / len(divergences) if divergences else 0.3
    # Normalize: stdev 0.15+ = high conflict
    norm_divergence = min(1.0, avg_divergence / 0.15)

    return 0.6 * contradiction_density + 0.4 * norm_divergence


def _compute_novelty(state: dict[str, Any], prev_entropy: EntropyState | None) -> float:
    """e_nov: new query ratio + outline heading delta."""
    sub_queries = state.get("sub_queries", [])
    follow_ups = state.get("follow_up_queries", [])

    # New query ratio: follow-ups / total queries
    total_queries = max(len(sub_queries), 1)
    new_ratio = len(follow_ups) / total_queries

    # If first iteration, high novelty
    if prev_entropy is None or prev_entropy["turn"] == 0:
        return 0.7

    return min(1.0, new_ratio * 2.0)  # scale: 50% new queries = max novelty


def _compute_trust(state: dict[str, Any]) -> float:
    """e_trust: grounding score variance + unmapped citations."""
    sections = state.get("section_drafts", [])

    if not sections:
        return 0.5

    # Grounding score variance
    grounding_scores = [s.get("grounding_score", 0.5) for s in sections if "grounding_score" in s]
    if len(grounding_scores) >= 2:
        try:
            grounding_var = statistics.variance(grounding_scores)
        except statistics.StatisticsError:
            grounding_var = 0.1
    else:
        grounding_var = 0.1

    # Normalize: variance 0.05+ = low trust
    trust_signal = min(1.0, grounding_var / 0.05)

    # Unmapped citation ratio
    citations = state.get("citations", [])
    passage_map = state.get("citation_to_passage_map", {})
    if citations:
        unmapped = sum(1 for c in citations if c.get("id", "") not in passage_map)
        unmapped_ratio = unmapped / len(citations)
    else:
        unmapped_ratio = 0.3  # default when no citations yet

    return 0.5 * trust_signal + 0.5 * unmapped_ratio

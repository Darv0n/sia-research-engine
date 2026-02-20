"""Confidence classification and convergence/replan logic."""

from __future__ import annotations

from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft


def classify_confidence(score: float) -> Confidence:
    """Classify a numeric score into confidence level."""
    if score >= 0.8:
        return Confidence.HIGH
    if score >= 0.6:
        return Confidence.MEDIUM
    return Confidence.LOW


def aggregate_grader_scores(scores: GraderScores) -> float:
    """Aggregate three grader scores into a single confidence score.

    Equal weighting across relevance, hallucination, and quality.
    """
    return (scores["relevance"] + scores["hallucination"] + scores["quality"]) / 3.0


def should_replan(
    section_drafts: list[SectionDraft],
    *,
    prev_avg: float = 0.0,
    delta_threshold: float = 0.05,
) -> tuple[bool, str]:
    """Determine whether to re-plan based on section confidence scores.

    Returns (should_replan, reason).

    Triggers replan if:
    - Any section has LOW confidence
    - More than 30% of sections are MEDIUM
    - (Does NOT replan if improvement delta < threshold — diminishing returns)
    """
    if not section_drafts:
        return True, "no_sections"

    scores = [s["confidence_score"] for s in section_drafts]
    levels = [s["confidence_level"] for s in section_drafts]
    avg = sum(scores) / len(scores)

    low_count = sum(1 for lv in levels if lv == Confidence.LOW)
    medium_count = sum(1 for lv in levels if lv == Confidence.MEDIUM)
    total = len(levels)

    if low_count > 0:
        return True, f"{low_count} section(s) with LOW confidence"

    medium_ratio = medium_count / total
    if medium_ratio > 0.3:
        return True, f"{medium_count}/{total} sections at MEDIUM confidence ({medium_ratio:.0%})"

    # Diminishing returns check
    if prev_avg > 0 and (avg - prev_avg) < delta_threshold:
        return False, f"diminishing_returns (delta={avg - prev_avg:.3f})"

    return False, "all_acceptable"


def summarize_confidence(section_drafts: list[SectionDraft]) -> dict[str, int]:
    """Count sections by confidence level."""
    counts: dict[str, int] = {c.value: 0 for c in Confidence}
    for s in section_drafts:
        level = s["confidence_level"]
        if isinstance(level, Confidence):
            counts[level.value] += 1
        else:
            counts[str(level)] += 1
    return counts

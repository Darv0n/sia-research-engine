"""Confidence heat map — Markdown table showing per-section confidence."""

from __future__ import annotations

from deep_research_swarm.contracts import Confidence, SectionDraft


def _confidence_indicator(level: Confidence | str) -> str:
    """Map confidence level to a text indicator."""
    level_str = level.value if isinstance(level, Confidence) else str(level)
    return {
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
        "LOW": "LOW",
    }.get(level_str, "???")


def render_confidence_heatmap(section_drafts: list[SectionDraft]) -> str:
    """Render a Markdown table of section confidence scores."""
    if not section_drafts:
        return "*No sections to assess.*"

    lines: list[str] = []
    lines.append("| Section | Confidence | Relevance | Hallucination | Quality |")
    lines.append("|---------|------------|-----------|---------------|---------|")

    for sec in section_drafts:
        heading = sec["heading"]
        level = _confidence_indicator(sec["confidence_level"])
        scores = sec["grader_scores"]
        lines.append(
            f"| {heading} | {level} ({sec['confidence_score']:.2f}) "
            f"| {scores['relevance']:.2f} "
            f"| {scores['hallucination']:.2f} "
            f"| {scores['quality']:.2f} |"
        )

    return "\n".join(lines)

"""Markdown report generation with YAML frontmatter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from deep_research_swarm.graph.state import ResearchState

from deep_research_swarm.reporting.citations import (
    build_bibliography,
    deduplicate_and_renumber,
)
from deep_research_swarm.reporting.heatmap import render_confidence_heatmap


def render_report(state: "ResearchState") -> str:
    """Render a full Markdown research report from graph state."""
    research_question = state["research_question"]
    raw_sections = state.get("section_drafts", [])
    raw_citations = state.get("citations", [])

    # Deduplicate and renumber citations for clean output
    section_drafts, citations = deduplicate_and_renumber(raw_sections, raw_citations)
    research_gaps = state.get("research_gaps", [])
    iteration_history = state.get("iteration_history", [])
    total_cost = state.get("total_cost_usd", 0.0)
    total_tokens = state.get("total_tokens_used", 0)
    convergence_reason = state.get("convergence_reason", "")

    # --- YAML Frontmatter ---
    frontmatter = {
        "title": f"Research Report: {research_question}",
        "generated": datetime.now(timezone.utc).isoformat(),
        "iterations": len(iteration_history),
        "total_sections": len(section_drafts),
        "total_citations": len(citations),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "convergence_reason": convergence_reason,
    }

    lines: list[str] = []
    lines.append("---")
    lines.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip())
    lines.append("---")
    lines.append("")

    # --- Title ---
    lines.append(f"# {research_question}")
    lines.append("")

    # --- Sections ---
    for sec in section_drafts:
        lines.append(f"## {sec['heading']}")
        lines.append("")
        lines.append(sec["content"])
        lines.append("")

    # --- Confidence Assessment ---
    if section_drafts:
        lines.append("## Confidence Assessment")
        lines.append("")
        lines.append(render_confidence_heatmap(section_drafts))
        lines.append("")

    # --- Research Gaps ---
    if research_gaps:
        lines.append("## Research Gaps")
        lines.append("")
        for gap in research_gaps:
            reason_label = {
                "no_sources": "No sources found",
                "contradictory": "Contradictory evidence",
                "low_confidence": "Low confidence",
            }.get(gap["reason"], gap["reason"])
            lines.append(f"- **{reason_label}**: {gap['description']}")
        lines.append("")

    # --- Bibliography ---
    if citations:
        lines.append("## Bibliography")
        lines.append("")
        lines.append(build_bibliography(citations))
        lines.append("")

    # --- Metadata ---
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated in {len(iteration_history)} iteration(s) | "
        f"{total_tokens:,} tokens | ${total_cost:.4f} USD*"
    )

    return "\n".join(lines)

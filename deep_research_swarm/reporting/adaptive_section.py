"""Adaptive adjustments report section (V8, PR-11).

Renders a Markdown section showing what the adaptive overseer adjusted
during the research run, including complexity profile and tunable changes.
"""

from __future__ import annotations

from deep_research_swarm.contracts import AdaptationEvent, ComplexityProfile


def render_adaptive_section(
    adaptation_events: list[AdaptationEvent],
    complexity_profile: ComplexityProfile | dict | None = None,
) -> str:
    """Render an "Adaptive Adjustments" Markdown section.

    Shows complexity multiplier, then a table of all tunable adjustments.
    Returns empty string if no adaptive data present.
    """
    if not adaptation_events and not complexity_profile:
        return ""

    lines: list[str] = []
    lines.append("## Adaptive Adjustments")
    lines.append("")

    # Complexity profile summary
    if complexity_profile:
        multiplier = complexity_profile.get("multiplier", 1.0)
        result_count = complexity_profile.get("result_count", 0)
        backends_used = complexity_profile.get("backends_used", 0)
        lines.append(
            f"**Complexity multiplier**: {multiplier:.2f} "
            f"({result_count} results, {backends_used} backends)"
        )
        lines.append("")

    # Adaptation events table
    if adaptation_events:
        lines.append("| Tunable | Old | New | Trigger | Reason |")
        lines.append("|---------|-----|-----|---------|--------|")

        for event in adaptation_events:
            name = event.get("tunable_name", "")
            old_val = event.get("old_value", "")
            new_val = event.get("new_value", "")
            trigger = event.get("trigger", "")
            reason = event.get("reason", "")

            # Format numbers nicely
            old_str = _fmt_val(old_val)
            new_str = _fmt_val(new_val)

            lines.append(f"| {name} | {old_str} | {new_str} | {trigger} | {reason} |")

        lines.append("")

    return "\n".join(lines)


def _fmt_val(val: int | float | str) -> str:
    """Format a tunable value for display."""
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)

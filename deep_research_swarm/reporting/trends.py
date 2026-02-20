"""Confidence trend visualization across iterations with unicode sparklines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deep_research_swarm.contracts import IterationRecord

# Sparkline characters from lowest to highest
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """Render a list of 0-1 floats as a unicode sparkline."""
    if not values:
        return ""
    chars = []
    for v in values:
        clamped = max(0.0, min(1.0, v))
        idx = int(clamped * (len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def render_confidence_trends(iteration_history: list[IterationRecord]) -> str:
    """Render a per-section confidence trend table across iterations.

    Returns a Markdown table with sparkline visualizations.
    Only meaningful with >= 2 iterations; returns empty string for < 2.
    """
    if len(iteration_history) < 2:
        return ""

    # Collect per-heading confidence across iterations
    # heading -> [score_iter1, score_iter2, ...]
    heading_scores: dict[str, list[float | None]] = {}
    num_iters = len(iteration_history)

    for iter_idx, record in enumerate(iteration_history):
        snapshots = record.get("section_snapshots", [])
        seen_headings: set[str] = set()

        for snap in snapshots:
            heading = snap["heading"]
            seen_headings.add(heading)

            if heading not in heading_scores:
                # Backfill None for previous iterations where this section didn't exist
                heading_scores[heading] = [None] * iter_idx

            heading_scores[heading].append(snap["confidence_score"])

        # For headings that existed before but aren't in this iteration
        for heading in heading_scores:
            if heading not in seen_headings:
                heading_scores[heading].append(None)

    # Pad any short lists to num_iters length
    for heading in heading_scores:
        while len(heading_scores[heading]) < num_iters:
            heading_scores[heading].append(None)

    if not heading_scores:
        return ""

    # Build iteration column headers
    iter_headers = " | ".join(f"Iter {i + 1}" for i in range(num_iters))
    iter_separators = " | ".join("------" for _ in range(num_iters))

    lines: list[str] = []
    lines.append(f"| Section | {iter_headers} | Trend |")
    lines.append(f"|---------|{iter_separators}|-------|")

    for heading, scores in heading_scores.items():
        cells = []
        sparkline_values = []
        for s in scores:
            if s is not None:
                cells.append(f"{s:.2f}")
                sparkline_values.append(s)
            else:
                cells.append("—")
        row = " | ".join(cells)
        spark = _sparkline(sparkline_values)
        lines.append(f"| {heading} | {row} | {spark} |")

    return "\n".join(lines)

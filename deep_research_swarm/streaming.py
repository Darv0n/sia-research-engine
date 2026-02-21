"""Streaming display for real-time progress during graph execution."""

from __future__ import annotations

import sys
from typing import Any

# Human-readable labels for graph node names
NODE_LABELS: dict[str, str] = {
    "health_check": "Checking backends",
    "plan": "Planning research",
    "search": "Searching sources",
    "extract": "Extracting content",
    "score": "Scoring documents",
    "synthesize": "Synthesizing report",
    "critique": "Critiquing sections",
    "rollup_budget": "Rolling up budget",
    "report": "Generating report",
    "contradiction": "Detecting contradictions",
    "plan_gate": "Awaiting plan approval",
    "report_gate": "Awaiting report approval",
}


class StreamDisplay:
    """Handles stream events from LangGraph astream and prints progress to stderr."""

    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self._current_node: str | None = None
        self._iteration: int = 0

    def _print(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def handle_update(self, update: dict[str, Any]) -> None:
        """Handle an 'updates' stream event (node_name -> state_update)."""
        for node_name, state_delta in update.items():
            label = NODE_LABELS.get(node_name, node_name)

            # Detect iteration changes
            new_iter = self._detect_iteration(state_delta)
            if new_iter and new_iter != self._iteration:
                self._iteration = new_iter
                self._print(f"\n--- Iteration {self._iteration} ---")

            self._current_node = node_name
            self._print(
                f"  [{label}]",
            )

            if self._verbose:
                self._print_details(node_name, state_delta)

    def handle_custom(self, event: dict[str, Any]) -> None:
        """Handle a 'custom' stream event (granular progress from nodes)."""
        kind = event.get("kind", "")
        msg = event.get("message", "")
        count = event.get("count")

        if kind == "search_progress":
            count_str = f" ({count} results)" if count is not None else ""
            self._print(f"    search{count_str}: {msg}")
        elif kind == "extract_progress":
            count_str = f" ({count} extracted)" if count is not None else ""
            self._print(f"    extract{count_str}: {msg}")
        elif msg:
            self._print(f"    {msg}")

    def _detect_iteration(self, state_delta: dict) -> int | None:
        """Extract current_iteration from state delta if present."""
        return state_delta.get("current_iteration")

    def _print_details(self, node_name: str, state_delta: dict) -> None:
        """Print verbose details about what a node produced."""
        counts: dict[str, str] = {}
        if "search_results" in state_delta:
            counts["results"] = str(len(state_delta["search_results"]))
        if "extracted_contents" in state_delta:
            counts["extracted"] = str(len(state_delta["extracted_contents"]))
        if "scored_documents" in state_delta:
            counts["scored"] = str(len(state_delta["scored_documents"]))
        if "section_drafts" in state_delta:
            counts["sections"] = str(len(state_delta["section_drafts"]))
        if "sub_queries" in state_delta:
            counts["queries"] = str(len(state_delta["sub_queries"]))

        if counts:
            detail = ", ".join(f"{k}={v}" for k, v in counts.items())
            self._print(f"    -> {detail}")

"""Extract a ResearchMemory record from completed graph state."""

from __future__ import annotations

from datetime import datetime, timezone

from deep_research_swarm.contracts import ResearchMemory


def extract_memory_record(state: dict, thread_id: str) -> ResearchMemory:
    """Deterministic extraction — no LLM call. Section headings only."""
    key_findings = [s["heading"] for s in state.get("section_drafts", []) if s.get("heading")]
    gaps = [g["description"] for g in state.get("research_gaps", []) if g.get("description")]
    sources_count = len({s["url"] for s in state.get("search_results", []) if s.get("url")})

    return ResearchMemory(
        thread_id=thread_id,
        question=state.get("research_question", ""),
        timestamp=datetime.now(timezone.utc).isoformat(),
        key_findings=key_findings,
        gaps=gaps,
        sources_count=sources_count,
        iterations=len(state.get("iteration_history", [])),
        converged=state.get("converged", False),
    )

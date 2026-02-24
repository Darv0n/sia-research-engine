"""ResearchState — the single state object flowing through the graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from deep_research_swarm.contracts import (
    AdaptationEvent,
    Citation,
    ComplexityProfile,
    Contradiction,
    DiversityMetrics,
    ExtractedContent,
    IterationRecord,
    ResearchGap,
    ScoredDocument,
    SearchResult,
    SectionDraft,
    SourcePassage,
    SubQuery,
    TokenUsage,
)

# --- Scalar reducers (last-write-wins) ---


def _replace(existing: str, new: str) -> str:
    return new


def _replace_int(existing: int, new: int) -> int:
    return new


def _replace_float(existing: float, new: float) -> float:
    return new


def _replace_bool(existing: bool, new: bool) -> bool:
    return new


def _replace_list(existing: list, new: list) -> list:
    """Replace-last-write for list fields (overwrites, not appends)."""
    return new


def _replace_dict(existing: dict, new: dict) -> dict:
    """Replace-last-write for dict fields."""
    return new


# --- Graph State ---


class ResearchState(TypedDict):
    # Input (set once)
    research_question: str
    max_iterations: Annotated[int, _replace_int]
    token_budget: Annotated[int, _replace_int]
    search_backends: Annotated[list[str], operator.add]
    memory_context: Annotated[str, _replace]

    # Planning (accumulate across iterations)
    perspectives: Annotated[list[str], operator.add]
    sub_queries: Annotated[list[SubQuery], operator.add]

    # Search + Extraction (accumulate across iterations)
    search_results: Annotated[list[SearchResult], operator.add]
    extracted_contents: Annotated[list[ExtractedContent], operator.add]

    # Scoring + Synthesis (replace each iteration — these are the "current" view)
    scored_documents: Annotated[list[ScoredDocument], _replace_list]
    diversity_metrics: Annotated[DiversityMetrics, _replace_dict]
    section_drafts: Annotated[list[SectionDraft], _replace_list]
    citations: Annotated[list[Citation], _replace_list]

    # Contradiction detection
    contradictions: Annotated[list[Contradiction], _replace_list]

    # Critique
    research_gaps: Annotated[list[ResearchGap], _replace_list]

    # Control
    current_iteration: Annotated[int, _replace_int]
    converged: Annotated[bool, _replace_bool]
    convergence_reason: Annotated[str, _replace]

    # Budget (accumulate)
    token_usage: Annotated[list[TokenUsage], operator.add]
    total_tokens_used: Annotated[int, _replace_int]
    total_cost_usd: Annotated[float, _replace_float]

    # History + Output
    iteration_history: Annotated[list[IterationRecord], operator.add]
    final_report: Annotated[str, _replace]

    # Execution mode (persisted for resume)
    mode: Annotated[str, _replace]

    # Passage grounding (V7) — accumulate across iterations
    source_passages: Annotated[list[SourcePassage], operator.add]
    citation_chain_results: Annotated[list[SearchResult], operator.add]

    # Citation-to-passage mapping (V7) — replace each iteration (OE4, D3)
    citation_to_passage_map: Annotated[dict[str, list[str]], _replace_dict]

    # Adaptive control (V8) — overseer state
    tunable_snapshot: Annotated[dict, _replace_dict]
    adaptation_events: Annotated[list[AdaptationEvent], operator.add]
    complexity_profile: Annotated[ComplexityProfile, _replace_dict]

    # Reactive search (V9) — gap analysis follow-up
    follow_up_queries: Annotated[list[SubQuery], operator.add]
    follow_up_round: Annotated[int, _replace_int]

    # Pre-research clarification (V9) — scope hints
    scope_hints: Annotated[dict, _replace_dict]

    # Entropy thermodynamics (V10/SIA) — convergence control
    entropy_state: Annotated[dict, _replace_dict]
    entropy_history: Annotated[list[dict], operator.add]

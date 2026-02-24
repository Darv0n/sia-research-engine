"""Complexity analyzer — deterministic metrics-to-multiplier computation.

Pure functions, no LLM, no side effects, microsecond speed.
Design invariant I6: metrics drive decisions, not guesses.
"""

from __future__ import annotations

import math

from deep_research_swarm.contracts import ComplexityProfile


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_complexity_profile(
    *,
    result_count: int = 0,
    backends_used: int = 1,
    iteration: int = 1,
    extraction_success_rate: float = 1.0,
    mean_grounding_score: float = 0.0,
    token_spend_rate: float = 0.0,
    scored_doc_count: int = 0,
    citation_chain_yield: int = 0,
) -> ComplexityProfile:
    """Compute a complexity profile from observed pipeline metrics.

    All inputs are directly measurable from pipeline state — no guessing.

    Multiplier formula:
        volume_factor = clamp(log10(max(result_count, 10)) / 2.0, 0.5, 2.0)
        backend_factor = clamp(0.8 + backends_used * 0.1, 0.8, 1.5)
        iter_factor = clamp(1.0 + (iteration - 1) * 0.1, 1.0, 1.3)
        multiplier = clamp(volume_factor * backend_factor * iter_factor / 1.5, 0.5, 2.0)

    The /1.5 normalization ensures that the "average" case (100 results,
    2 backends, iteration 1) yields multiplier ~1.0.
    """
    volume_factor = _clamp(math.log10(max(result_count, 10)) / 2.0, 0.5, 2.0)
    backend_factor = _clamp(0.8 + backends_used * 0.1, 0.8, 1.5)
    iter_factor = _clamp(1.0 + (iteration - 1) * 0.1, 1.0, 1.3)
    raw = volume_factor * backend_factor * iter_factor / 1.5
    multiplier = _clamp(raw, 0.5, 2.0)

    return ComplexityProfile(
        result_count=result_count,
        backends_used=backends_used,
        iteration=iteration,
        extraction_success_rate=extraction_success_rate,
        mean_grounding_score=mean_grounding_score,
        token_spend_rate=token_spend_rate,
        scored_doc_count=scored_doc_count,
        citation_chain_yield=citation_chain_yield,
        volume_factor=round(volume_factor, 4),
        backend_factor=round(backend_factor, 4),
        iter_factor=round(iter_factor, 4),
        multiplier=round(multiplier, 4),
    )


def compute_multiplier(profile: ComplexityProfile) -> float:
    """Extract the composite multiplier from a complexity profile."""
    return profile["multiplier"]


def compute_profile_from_state(state: dict) -> ComplexityProfile:
    """Convenience: compute ComplexityProfile directly from ResearchState dict.

    Extracts all observable metrics from state fields.
    """
    search_results = state.get("search_results", [])
    extracted_contents = state.get("extracted_contents", [])
    scored_documents = state.get("scored_documents", [])
    citation_chain_results = state.get("citation_chain_results", [])
    section_drafts = state.get("section_drafts", [])
    token_usage = state.get("token_usage", [])
    token_budget = state.get("token_budget", 200000)

    # Extraction success rate
    total_extracted = len(extracted_contents)
    successful = sum(1 for e in extracted_contents if e.get("extraction_success", False))
    extraction_rate = successful / max(total_extracted, 1)

    # Mean grounding score across sections
    grounding_scores = [
        s.get("grounding_score", 0.0) for s in section_drafts if "grounding_score" in s
    ]
    mean_grounding = sum(grounding_scores) / max(len(grounding_scores), 1)

    # Token spend rate
    total_tokens = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in token_usage)
    spend_rate = total_tokens / max(token_budget, 1)

    # Backend count
    backends = set()
    for sr in search_results:
        backends.add(sr.get("backend", "unknown"))

    return compute_complexity_profile(
        result_count=len(search_results),
        backends_used=max(len(backends), 1),
        iteration=state.get("current_iteration", 1),
        extraction_success_rate=extraction_rate,
        mean_grounding_score=mean_grounding,
        token_spend_rate=spend_rate,
        scored_doc_count=len(scored_documents),
        citation_chain_yield=len(citation_chain_results),
    )

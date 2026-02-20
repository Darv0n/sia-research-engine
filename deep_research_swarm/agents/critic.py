"""Critic agent — three-grader chain with convergence detection.

Three specialized graders evaluate distinct failure modes:
- Relevance: Does the content answer the research question?
- Hallucination: Are claims grounded in cited sources?
- Quality: Is the writing clear, deep, and well-organized?
"""

from __future__ import annotations

import asyncio

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import (
    GraderScores,
    IterationRecord,
    SectionConfidenceSnapshot,
    SectionDraft,
)
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.scoring.confidence import (
    classify_confidence,
    should_replan,
    summarize_confidence,
)

# --- Three separate grader prompts ---

RELEVANCE_SYSTEM = """\
You are a relevance grader. Evaluate how well each section answers \
the research question. Score from 0.0 to 1.0.

1.0 = directly answers the question with depth
0.7 = partially relevant but missing key aspects
0.4 = tangentially related
0.0 = completely off-topic

Output STRICT JSON:
{"scores": [{"section_id": "sec-xxx", "relevance": 0.85}]}
"""

HALLUCINATION_SYSTEM = """\
You are a hallucination grader. Evaluate whether claims in each \
section are grounded in the cited sources. Score from 0.0 to 1.0.

1.0 = every claim has a citation and matches source content
0.7 = most claims grounded, minor unsupported statements
0.4 = significant unsupported claims
0.0 = fabricated content with no source basis

Check: Are [N] citation markers present? Do claims match what \
sources would reasonably say? Flag speculation presented as fact.

Output STRICT JSON:
{"scores": [{"section_id": "sec-xxx", "hallucination": 0.90}]}
"""

QUALITY_SYSTEM = """\
You are a writing quality grader. Evaluate clarity, depth, \
organization, and usefulness of each section. Score from 0.0 to 1.0.

1.0 = excellent: clear structure, sufficient depth, useful insights
0.7 = good: readable but could be deeper or better organized
0.4 = mediocre: superficial, poorly structured, or repetitive
0.0 = unusable: incoherent or empty

Output STRICT JSON:
{"scores": [{"section_id": "sec-xxx", "quality": 0.80}]}
"""


async def _grade_dimension(
    caller: AgentCaller,
    system: str,
    sections_text: str,
    research_question: str,
    dimension: str,
    agent_name: str,
) -> tuple[dict[str, float], dict]:
    """Run a single grader dimension. Returns (id->score, usage)."""
    user_content = (
        f"Research question: {research_question}\n\nSections to evaluate:\n{sections_text}"
    )

    data, usage = await caller.call_json(
        system=system,
        messages=[{"role": "user", "content": user_content}],
        agent_name=agent_name,
        max_tokens=1024,
    )

    scores_by_id: dict[str, float] = {}
    for item in data.get("scores", []):
        sid = item.get("section_id", "")
        score = item.get(dimension, 0.5)
        scores_by_id[sid] = score

    return scores_by_id, usage


async def critique(state: ResearchState, caller: AgentCaller) -> dict:
    """Three-grader chain: evaluate sections then determine convergence."""
    research_question = state["research_question"]
    section_drafts = state.get("section_drafts", [])
    current_iteration = state.get("current_iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    total_tokens = state.get("total_tokens_used", 0)
    token_budget = state.get("token_budget", 200000)

    if not section_drafts:
        return {
            "converged": True,
            "convergence_reason": "no_sections_to_evaluate",
        }

    # Build section text shared by all three graders
    sections_text = ""
    for sec in section_drafts:
        sections_text += (
            f"\n--- Section: {sec['heading']} (id: {sec['id']}) ---\n{sec['content']}\n"
        )

    # Run three graders in parallel
    relevance_task = _grade_dimension(
        caller,
        RELEVANCE_SYSTEM,
        sections_text,
        research_question,
        "relevance",
        "critic_relevance",
    )
    hallucination_task = _grade_dimension(
        caller,
        HALLUCINATION_SYSTEM,
        sections_text,
        research_question,
        "hallucination",
        "critic_hallucination",
    )
    quality_task = _grade_dimension(
        caller,
        QUALITY_SYSTEM,
        sections_text,
        research_question,
        "quality",
        "critic_quality",
    )

    (
        (rel_scores, rel_usage),
        (hal_scores, hal_usage),
        (qual_scores, qual_usage),
    ) = await asyncio.gather(relevance_task, hallucination_task, quality_task)

    all_usages = [rel_usage, hal_usage, qual_usage]

    # Merge three dimensions into updated section drafts
    updated_sections: list[SectionDraft] = []
    for sec in section_drafts:
        sid = sec["id"]
        rel = rel_scores.get(sid, sec["grader_scores"]["relevance"])
        hal = hal_scores.get(sid, sec["grader_scores"]["hallucination"])
        qual = qual_scores.get(sid, sec["grader_scores"]["quality"])

        scores = GraderScores(relevance=rel, hallucination=hal, quality=qual)
        avg = (rel + hal + qual) / 3.0

        updated_sections.append(
            SectionDraft(
                id=sid,
                heading=sec["heading"],
                content=sec["content"],
                citation_ids=sec["citation_ids"],
                confidence_score=round(avg, 4),
                confidence_level=classify_confidence(avg),
                grader_scores=scores,
            )
        )

    # Determine convergence
    prev_history = state.get("iteration_history", [])
    prev_avg = prev_history[-1]["avg_confidence"] if prev_history else 0.0

    replan, reason = should_replan(updated_sections, prev_avg=prev_avg)

    # Force convergence conditions
    if current_iteration >= max_iterations:
        replan = False
        reason = f"max_iterations_reached ({max_iterations})"

    if total_tokens > token_budget * 0.9:
        replan = False
        reason = f"budget_nearly_exhausted ({total_tokens}/{token_budget})"

    converged = not replan

    # Build iteration record
    scores_list = [s["confidence_score"] for s in updated_sections]
    avg_conf = sum(scores_list) / len(scores_list) if scores_list else 0.0

    section_snapshots = [
        SectionConfidenceSnapshot(
            heading=s["heading"],
            confidence_score=s["confidence_score"],
            confidence_level=(
                s["confidence_level"].value
                if hasattr(s["confidence_level"], "value")
                else str(s["confidence_level"])
            ),
        )
        for s in updated_sections
    ]

    iteration_record = IterationRecord(
        iteration=current_iteration,
        sub_queries_generated=len(state.get("sub_queries", [])),
        search_results_found=len(state.get("search_results", [])),
        documents_extracted=len(state.get("extracted_contents", [])),
        sections_drafted=len(updated_sections),
        avg_confidence=round(avg_conf, 4),
        sections_by_confidence=summarize_confidence(updated_sections),
        token_usage=all_usages,
        replan_reason=None if converged else reason,
        section_snapshots=section_snapshots,
    )

    return {
        "section_drafts": updated_sections,
        "converged": converged,
        "convergence_reason": reason,
        "iteration_history": [iteration_record],
        "token_usage": all_usages,
    }

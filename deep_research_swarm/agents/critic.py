"""Critic agent — quality grading with convergence detection."""

from __future__ import annotations

import json

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.scoring.confidence import (
    classify_confidence,
    should_replan,
    summarize_confidence,
)

CRITIC_SYSTEM = """\
You are a research quality critic. Evaluate each section of a research synthesis \
for quality, relevance, and potential hallucination.

For each section, provide scores from 0.0 to 1.0:
- relevance: How well the section answers the research question (1.0 = perfectly relevant)
- hallucination: Confidence that claims are grounded in sources (1.0 = fully grounded, no hallucination)
- quality: Overall writing quality, depth, and usefulness (1.0 = excellent)

Output STRICT JSON (no markdown, no commentary):
{
  "evaluations": [
    {
      "section_id": "sec-xxx",
      "relevance": 0.85,
      "hallucination": 0.90,
      "quality": 0.80
    }
  ]
}

Rules:
- Be critical but fair. Most sections should score between 0.5 and 0.9.
- Flag hallucination (low score) if claims lack citations or contradict sources.
- Flag low quality if the section is superficial or poorly organized.
"""


async def critique(state: ResearchState, caller: AgentCaller) -> dict:
    """Evaluate section drafts and determine convergence."""
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

    # Build section text for evaluation
    sections_text = ""
    for sec in section_drafts:
        sections_text += (
            f"\n--- Section: {sec['heading']} (id: {sec['id']}) ---\n"
            f"{sec['content']}\n"
        )

    user_content = (
        f"Research question: {research_question}\n\n"
        f"Sections to evaluate:\n{sections_text}"
    )

    data, usage = await caller.call_json(
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="critic",
        max_tokens=2048,
    )

    # Update section drafts with grader scores
    eval_by_id: dict[str, dict] = {}
    for ev in data.get("evaluations", []):
        eval_by_id[ev["section_id"]] = ev

    updated_sections: list[SectionDraft] = []
    for sec in section_drafts:
        ev = eval_by_id.get(sec["id"])
        if ev:
            scores = GraderScores(
                relevance=ev.get("relevance", sec["grader_scores"]["relevance"]),
                hallucination=ev.get("hallucination", sec["grader_scores"]["hallucination"]),
                quality=ev.get("quality", sec["grader_scores"]["quality"]),
            )
            avg = (scores["relevance"] + scores["hallucination"] + scores["quality"]) / 3.0
            sec = SectionDraft(
                id=sec["id"],
                heading=sec["heading"],
                content=sec["content"],
                citation_ids=sec["citation_ids"],
                confidence_score=round(avg, 4),
                confidence_level=classify_confidence(avg),
                grader_scores=scores,
            )
        updated_sections.append(sec)

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

    from deep_research_swarm.contracts import IterationRecord

    iteration_record = IterationRecord(
        iteration=current_iteration,
        sub_queries_generated=len(state.get("sub_queries", [])),
        search_results_found=len(state.get("search_results", [])),
        documents_extracted=len(state.get("extracted_contents", [])),
        sections_drafted=len(updated_sections),
        avg_confidence=round(avg_conf, 4),
        sections_by_confidence=summarize_confidence(updated_sections),
        token_usage=[usage],
        replan_reason=None if converged else reason,
    )

    return {
        "section_drafts": updated_sections,
        "converged": converged,
        "convergence_reason": reason,
        "iteration_history": [iteration_record],
        "token_usage": [usage],
    }

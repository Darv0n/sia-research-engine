"""Planner agent — STORM-style decomposition with query deduplication."""

from __future__ import annotations

import json
import uuid

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import SubQuery
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.utils.text import is_duplicate as _is_duplicate

PLANNER_SYSTEM = """\
You are a research planning agent. Given a research question, decompose it \
using STORM-style perspective-guided questioning.

Your job:
1. Identify 3 diverse perspectives from which to investigate the question.
2. For each perspective, generate 1-2 specific sub-queries for a search engine.
3. Assign priority (1=highest, 5=lowest) based on centrality to the question.

Output STRICT JSON (no markdown, no commentary):
{
  "perspectives": ["perspective1", "perspective2", "perspective3"],
  "sub_queries": [
    {
      "question": "specific search query",
      "perspective": "which perspective this serves",
      "priority": 1
    }
  ]
}

Rules:
- Sub-queries must be concrete and searchable (not vague or philosophical).
- Each sub-query should target a different facet of the question.
- If re-planning, you will see previous queries and gaps. \
Generate NEW queries that address the gaps. Do NOT repeat previous queries.
"""


async def plan(state: ResearchState, caller: AgentCaller) -> dict:
    """Decompose research question into perspectives and sub-queries."""
    research_question = state["research_question"]
    iteration = state.get("current_iteration", 0)
    existing_queries = [sq["question"] for sq in state.get("sub_queries", [])]
    gaps = state.get("research_gaps", [])

    user_content = f"Research question: {research_question}"

    memory_context = state.get("memory_context", "")
    if memory_context:
        user_content += (
            f"\n\nPrior research findings (use as background, do not repeat these queries):\n"
            f"{memory_context}"
        )

    if iteration > 0 and (existing_queries or gaps):
        user_content += f"\n\nIteration: {iteration + 1}"
        if existing_queries:
            user_content += (
                f"\nPrevious queries (do NOT repeat):\n{json.dumps(existing_queries, indent=2)}"
            )
        if gaps:
            gap_descs = [g["description"] for g in gaps]
            user_content += f"\nIdentified gaps to address:\n{json.dumps(gap_descs, indent=2)}"

    data, usage = await caller.call_json(
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="planner",
        max_tokens=2048,
    )

    perspectives = data.get("perspectives", [])
    backends = state.get("search_backends", ["searxng"])

    # Build sub-queries with programmatic deduplication
    sub_queries: list[SubQuery] = []
    accepted_questions = list(existing_queries)  # Track for within-batch dedup

    for sq in data.get("sub_queries", []):
        question = sq.get("question", "").strip()
        if not question:
            continue

        # Reject if duplicate of existing or already-accepted
        if _is_duplicate(question, accepted_questions):
            continue

        sub_queries.append(
            SubQuery(
                id=f"sq-{uuid.uuid4().hex[:8]}",
                question=question,
                perspective=sq.get("perspective", ""),
                priority=sq.get("priority", 3),
                parent_query_id=None,
                search_backends=backends,
            )
        )
        accepted_questions.append(question)

    return {
        "perspectives": perspectives,
        "sub_queries": sub_queries,
        "current_iteration": iteration + 1,
        "token_usage": [usage],
    }

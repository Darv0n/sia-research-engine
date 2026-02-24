"""Gap analyzer — reactive search within a single iteration (V9, G2).

Examines extracted content after initial search to identify knowledge gaps,
then generates targeted follow-up queries. This is a tighter loop than the
quality iteration loop (plan -> search -> ... -> critique -> re-plan), which
operates across iterations.

The gap_analysis node runs between score and adapt_synthesis. It can trigger
a follow-up search round (search_followup -> extract_followup -> score_merge)
or pass through to adapt_synthesis if no gaps found or budget exhausted.
"""

from __future__ import annotations

import json
import uuid

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import SubQuery
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.scoring.routing import classify_query, route_backends
from deep_research_swarm.utils.text import is_duplicate as _is_duplicate

GAP_ANALYSIS_SYSTEM = """\
You are a research gap analyzer. Given a research question and the content \
extracted so far, identify what important aspects are MISSING or UNDEREXPLORED.

Your job:
1. Review the research question and the evidence gathered so far.
2. Identify 3-5 specific knowledge gaps — areas where:
   - Key claims lack supporting evidence
   - Important perspectives are unrepresented
   - Contradictions need resolution with additional sources
   - Recent developments haven't been captured
   - Quantitative data is missing where it would strengthen the argument
3. For each gap, generate a targeted search query to fill it.

Output STRICT JSON (no markdown, no commentary):
{
  "gaps_found": true,
  "gaps": [
    {
      "description": "What is missing",
      "query": "targeted search query to fill this gap",
      "priority": 1
    }
  ]
}

If the evidence is comprehensive and no significant gaps exist, output:
{"gaps_found": false, "gaps": []}

Rules:
- Queries must be concrete and searchable.
- Do NOT repeat queries already executed.
- Focus on the most impactful gaps, not trivial ones.
- Maximum 5 follow-up queries.
"""


async def analyze_gaps(
    state: ResearchState,
    caller: AgentCaller,
    *,
    available_backends: list[str] | None = None,
    max_follow_up_queries: int = 5,
) -> dict:
    """Identify knowledge gaps and generate follow-up queries.

    Returns dict with:
      - follow_up_queries: list[SubQuery] (may be empty)
      - token_usage: [usage]
    """
    research_question = state["research_question"]
    scored_docs = state.get("scored_documents", [])
    existing_queries = [sq["question"] for sq in state.get("sub_queries", [])]
    follow_up_round = state.get("follow_up_round", 0)

    # Read adaptive tunable (V9) — follow-up budget
    _snap = state.get("tunable_snapshot", {})
    follow_up_budget = int(_snap.get("follow_up_budget", 5))

    # Don't run gap analysis if follow-up round already happened this iteration
    if follow_up_round > 0:
        return {"follow_up_queries": []}

    # Build context from top scored documents
    if not scored_docs:
        return {"follow_up_queries": []}

    top_docs = sorted(scored_docs, key=lambda d: d["combined_score"], reverse=True)[:15]
    evidence_summary = []
    for i, doc in enumerate(top_docs, 1):
        preview = doc["content"][:500]
        evidence_summary.append(f"[{i}] {doc['title']}\n{preview}")

    user_content = (
        f"Research question: {research_question}\n\n"
        f"Evidence gathered ({len(scored_docs)} documents scored, "
        f"showing top {len(top_docs)}):\n\n"
        + "\n\n".join(evidence_summary)
        + f"\n\nPrevious queries (do NOT repeat):\n{json.dumps(existing_queries, indent=2)}"
    )

    data, usage = await caller.call_json(
        system=GAP_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="gap_analyzer",
        max_tokens=2048,
    )

    if not data.get("gaps_found", False):
        return {
            "follow_up_queries": [],
            "follow_up_round": 1,  # Mark as completed
            "token_usage": [usage],
        }

    # Build follow-up sub-queries
    user_backends = state.get("search_backends", [])
    available_backends = available_backends or ["searxng"]

    follow_up_queries: list[SubQuery] = []
    accepted = list(existing_queries)
    effective_budget = min(follow_up_budget, max_follow_up_queries)

    for gap in data.get("gaps", [])[:effective_budget]:
        question = gap.get("query", "").strip()
        if not question:
            continue

        if _is_duplicate(question, accepted):
            continue

        if user_backends:
            backends = user_backends
        else:
            query_type = classify_query(question)
            backends = route_backends(query_type, available_backends)

        follow_up_queries.append(
            SubQuery(
                id=f"sq-fu-{uuid.uuid4().hex[:8]}",
                question=question,
                perspective="follow-up",
                priority=gap.get("priority", 2),
                parent_query_id=None,
                search_backends=backends,
            )
        )
        accepted.append(question)

    return {
        "follow_up_queries": follow_up_queries,
        "sub_queries": follow_up_queries,  # Also add to main sub_queries for history
        "follow_up_round": 1,
        "token_usage": [usage],
    }

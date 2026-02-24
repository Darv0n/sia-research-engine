"""Clarifier agent — pre-research scope analysis (V9, G5).

In HITL mode: generates clarifying questions and interrupts for user input.
In auto mode: infers scope hints from question structure (no LLM call).

The clarify node runs between health_check and plan. It annotates the state
with scope_hints that the planner uses to calibrate query generation.
"""

from __future__ import annotations

from typing import Any

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.graph.state import ResearchState

CLARIFIER_SYSTEM = """\
You are a research scope analyst. Given a research question, generate 2-3 \
clarifying questions that would help focus the research.

Focus on:
1. **Scope**: How broad or narrow should the research be?
2. **Depth**: Academic rigor vs practical overview?
3. **Recency**: Historical context vs current state?
4. **Audience**: Who will read this report?

Output STRICT JSON (no markdown, no commentary):
{
  "scope_hints": {
    "breadth": "narrow|moderate|broad",
    "depth": "overview|detailed|comprehensive",
    "recency": "historical|balanced|recent",
    "domain": "academic|technical|general|policy"
  },
  "clarifying_questions": [
    "Question about scope or focus?"
  ]
}
"""


def _infer_scope_hints(question: str) -> dict[str, str]:
    """Infer scope hints from question structure without LLM (auto mode).

    Uses keyword heuristics to classify question type.
    """
    q = question.lower()

    # Breadth
    if any(w in q for w in ["comprehensive", "overview", "survey", "landscape", "all"]):
        breadth = "broad"
    elif any(w in q for w in ["specific", "exactly", "particular", "single", "one"]):
        breadth = "narrow"
    else:
        breadth = "moderate"

    # Depth
    depth_words = [
        "academic",
        "scholarly",
        "peer-reviewed",
        "literature review",
        "meta-analysis",
        "studies",
    ]
    if any(w in q for w in depth_words):
        depth = "comprehensive"
    elif any(w in q for w in ["quick", "brief", "summary", "tldr", "simple", "overview"]):
        depth = "overview"
    else:
        depth = "detailed"

    # Recency
    if any(w in q for w in ["latest", "recent", "2025", "2026", "current", "new", "emerging"]):
        recency = "recent"
    elif any(w in q for w in ["history", "historical", "evolution", "origin", "timeline"]):
        recency = "historical"
    else:
        recency = "balanced"

    # Domain
    domain_words = [
        "paper",
        "study",
        "studies",
        "research",
        "journal",
        "doi",
        "arxiv",
        "peer-reviewed",
        "scholarly",
    ]
    if any(w in q for w in domain_words):
        domain = "academic"
    elif any(w in q for w in ["api", "code", "implementation", "framework", "library", "tool"]):
        domain = "technical"
    elif any(w in q for w in ["policy", "regulation", "law", "government", "legislation"]):
        domain = "policy"
    else:
        domain = "general"

    return {
        "breadth": breadth,
        "depth": depth,
        "recency": recency,
        "domain": domain,
    }


async def clarify(
    state: ResearchState,
    caller: AgentCaller | None = None,
    *,
    mode: str = "auto",
) -> dict:
    """Analyze research question scope and generate scope hints.

    In auto mode: uses keyword heuristics (no LLM call, no token cost).
    In HITL mode: uses LLM to generate clarifying questions (interrupted by gate).
    """
    question = state["research_question"]

    if mode == "auto" or caller is None:
        # Auto mode: deterministic inference, no LLM
        hints = _infer_scope_hints(question)
        return {"scope_hints": hints}

    # HITL mode: use LLM for richer analysis
    data, usage = await caller.call_json(
        system=CLARIFIER_SYSTEM,
        messages=[{"role": "user", "content": f"Research question: {question}"}],
        agent_name="clarifier",
        max_tokens=1024,
    )

    hints = data.get("scope_hints", _infer_scope_hints(question))
    questions = data.get("clarifying_questions", [])

    result: dict[str, Any] = {
        "scope_hints": hints,
        "token_usage": [usage],
    }

    # Store questions for the HITL gate to display
    if questions:
        result["_clarifying_questions"] = questions

    return result

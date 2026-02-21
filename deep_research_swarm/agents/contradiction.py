"""Contradiction detection agent — identifies conflicting claims between sources."""

from __future__ import annotations

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import Contradiction, ScoredDocument

CONTRADICTION_SYSTEM = """\
You are a contradiction detector. Given a set of source documents, identify \
any contradictions between them. Focus on factual claims that directly conflict.

Severity levels:
- "direct": Sources make mutually exclusive factual claims.
- "nuanced": Sources present different conclusions from similar data.
- "contextual": Sources appear to disagree but are addressing different contexts.

Examine each pair of sources for conflicting claims. Only report genuine \
contradictions, not differences in scope or emphasis.

Output STRICT JSON:
{
  "contradictions": [
    {
      "claim_a": "Specific claim from source A",
      "source_a_index": 1,
      "claim_b": "Conflicting claim from source B",
      "source_b_index": 2,
      "topic": "Brief topic description",
      "severity": "direct"
    }
  ]
}

If no contradictions are found, return: {"contradictions": []}

CRITICAL: Output ONLY the JSON object. No explanations, no analysis, no prose. \
Start your response with { and end with }."""


async def detect_contradictions(
    scored_documents: list[ScoredDocument],
    caller: AgentCaller,
    *,
    max_docs: int = 10,
) -> tuple[list[Contradiction], dict]:
    """Detect contradictions among the top scored documents.

    Returns (list of Contradiction dicts, token_usage).
    """
    if len(scored_documents) < 2:
        return [], {}

    top_docs = sorted(scored_documents, key=lambda d: d["combined_score"], reverse=True)[:max_docs]

    # Build source context
    sources_text = ""
    for i, doc in enumerate(top_docs, start=1):
        content_preview = doc["content"][:2000]
        sources_text += (
            f"\n--- Source [{i}] ---\n"
            f"Title: {doc['title']}\n"
            f"URL: {doc['url']}\n"
            f"Content:\n{content_preview}\n"
        )

    user_content = f"Analyze these {len(top_docs)} sources for contradictions:\n{sources_text}"

    data, usage = await caller.call_json(
        system=CONTRADICTION_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="contradiction_detector",
        max_tokens=8192,
    )

    contradictions: list[Contradiction] = []
    for item in data.get("contradictions", []):
        # Map source indices back to actual document URLs
        idx_a = item.get("source_a_index", 0)
        idx_b = item.get("source_b_index", 0)

        url_a = _safe_url(top_docs, idx_a)
        url_b = _safe_url(top_docs, idx_b)

        severity = item.get("severity", "contextual")
        if severity not in ("direct", "nuanced", "contextual"):
            severity = "contextual"

        contradictions.append(
            Contradiction(
                claim_a=item.get("claim_a", ""),
                source_a_url=url_a,
                claim_b=item.get("claim_b", ""),
                source_b_url=url_b,
                topic=item.get("topic", ""),
                severity=severity,
            )
        )

    return contradictions, usage


def _safe_url(docs: list[ScoredDocument], index: int) -> str:
    """Safely get URL from 1-based index, returning empty string for out-of-range."""
    if 1 <= index <= len(docs):
        return docs[index - 1]["url"]
    return ""

"""Synthesizer agent — RAG-Fusion synthesis with inline citations.

V2: Context-aware — on re-iterations, receives previous section drafts
and research gaps, refining rather than regenerating from scratch.
"""

from __future__ import annotations

import uuid

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    Contradiction,
    GraderScores,
    ResearchGap,
    ScoredDocument,
    SectionDraft,
)
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.scoring.confidence import classify_confidence

SYNTHESIZER_SYSTEM_INITIAL = """\
You are a research synthesizer. Given a research question and source \
documents, produce a structured synthesis with inline citations.

Your job:
1. Identify key themes and organize into sections.
2. Synthesize information from sources with inline [N] citations.
3. Assess your confidence in each section (0.0 to 1.0).

Output STRICT JSON:
{
  "sections": [
    {
      "heading": "Section Title",
      "content": "Synthesized content with [1] citations [2].",
      "source_ids": [1, 2],
      "confidence": 0.85
    }
  ],
  "gaps": [
    {"description": "What is missing", "reason": "no_sources"}
  ]
}

Rules:
- Every factual claim MUST have a [N] citation referencing source number.
- Confidence reflects source quality, agreement, and coverage.
- Gaps reasons: "no_sources", "contradictory", "low_confidence"
"""

SYNTHESIZER_SYSTEM_REFINE = """\
You are a research synthesizer refining an existing report with new data.

You have:
1. Previous sections (with confidence scores and identified weaknesses)
2. New source documents found to address gaps

Your job:
- KEEP strong sections (HIGH confidence) mostly intact, adding new \
citations only if new sources strengthen them.
- REVISE weak sections (MEDIUM/LOW confidence) using new sources.
- ADD new sections if new sources cover topics not yet addressed.
- REMOVE sections only if they were entirely wrong (rare).
- Address the listed research gaps where new sources help.

Output STRICT JSON:
{
  "sections": [
    {
      "heading": "Section Title",
      "content": "Revised content with [N] citations.",
      "source_ids": [1, 3],
      "confidence": 0.90,
      "action": "revised"
    }
  ],
  "gaps": [
    {"description": "Still missing", "reason": "no_sources"}
  ]
}

"action" must be one of: "kept", "revised", "new", "removed"

Rules:
- Every factual claim MUST have a [N] citation.
- Do NOT regenerate content that was already HIGH confidence.
- Focus effort on gaps and weaknesses from the previous iteration.
"""


def _build_source_context(
    scored_docs: list[ScoredDocument],
    max_docs: int = 15,
) -> tuple[str, dict[int, ScoredDocument]]:
    """Build numbered source context for the LLM."""
    top_docs = sorted(scored_docs, key=lambda d: d["combined_score"], reverse=True)[:max_docs]

    doc_index: dict[int, ScoredDocument] = {}
    sources_text = ""
    for i, doc in enumerate(top_docs, start=1):
        doc_index[i] = doc
        content_preview = doc["content"][:2000]
        sources_text += (
            f"\n--- Source [{i}] ---\n"
            f"Title: {doc['title']}\n"
            f"URL: {doc['url']}\n"
            f"Authority: {doc['authority']}\n"
            f"Score: {doc['combined_score']}\n"
            f"Content:\n{content_preview}\n"
        )

    return sources_text, doc_index


def _build_previous_context(
    section_drafts: list[SectionDraft],
    research_gaps: list[ResearchGap],
) -> str:
    """Build summary of previous iteration for refinement prompt."""
    parts: list[str] = []

    parts.append("=== PREVIOUS SECTIONS ===")
    for sec in section_drafts:
        level = sec["confidence_level"]
        if isinstance(level, Confidence):
            level = level.value
        parts.append(
            f"\n--- {sec['heading']} "
            f"[{level} {sec['confidence_score']:.2f}] ---\n"
            f"{sec['content'][:1000]}"
        )

    if research_gaps:
        parts.append("\n=== RESEARCH GAPS TO ADDRESS ===")
        for gap in research_gaps:
            parts.append(f"- [{gap['reason']}] {gap['description']}")

    return "\n".join(parts)


def _build_citations(
    data: dict,
    doc_index: dict[int, ScoredDocument],
    top_docs: list[ScoredDocument],
) -> tuple[list[SectionDraft], list[Citation], list[ResearchGap]]:
    """Parse LLM response into section drafts, citations, and gaps."""
    section_drafts: list[SectionDraft] = []
    all_citations: list[Citation] = []
    citation_counter = 0

    for sec in data.get("sections", []):
        conf_score = sec.get("confidence", 0.5)
        conf_level = classify_confidence(conf_score)

        source_ids = sec.get("source_ids", [])
        section_citation_ids: list[str] = []

        for src_ref in source_ids:
            citation_counter += 1
            cit_id = f"[{citation_counter}]"

            doc = None
            if isinstance(src_ref, int) and src_ref in doc_index:
                doc = doc_index[src_ref]
            elif isinstance(src_ref, str):
                for d in top_docs:
                    if d["id"] == src_ref:
                        doc = d
                        break

            if doc:
                all_citations.append(
                    Citation(
                        id=cit_id,
                        url=doc["url"],
                        title=doc["title"],
                        authority=doc["authority"],
                        accessed="",
                        used_in_sections=[sec["heading"]],
                    )
                )
                section_citation_ids.append(cit_id)

        section_drafts.append(
            SectionDraft(
                id=f"sec-{uuid.uuid4().hex[:8]}",
                heading=sec["heading"],
                content=sec["content"],
                citation_ids=section_citation_ids,
                confidence_score=round(conf_score, 4),
                confidence_level=conf_level,
                grader_scores=GraderScores(
                    relevance=conf_score,
                    hallucination=1.0,
                    quality=conf_score,
                ),
            )
        )

    research_gaps: list[ResearchGap] = []
    for gap in data.get("gaps", []):
        research_gaps.append(
            ResearchGap(
                description=gap.get("description", ""),
                attempted_queries=[],
                reason=gap.get("reason", "no_sources"),
            )
        )

    return section_drafts, all_citations, research_gaps


def _build_contradiction_context(contradictions: list[Contradiction]) -> str:
    """Build a summary of detected contradictions for the synthesizer."""
    if not contradictions:
        return ""

    parts = ["\n=== DETECTED CONTRADICTIONS ==="]
    parts.append("Address these contradictions in your synthesis:")
    for i, c in enumerate(contradictions, 1):
        severity_label = {
            "direct": "DIRECT CONFLICT",
            "nuanced": "NUANCED DIFFERENCE",
            "contextual": "CONTEXTUAL DIFFERENCE",
        }.get(c["severity"], c["severity"])
        parts.append(
            f"\n{i}. [{severity_label}] {c['topic']}\n"
            f"   Claim A: {c['claim_a']}\n"
            f"   Claim B: {c['claim_b']}"
        )
    return "\n".join(parts)


async def synthesize(state: ResearchState, caller: AgentCaller) -> dict:
    """Synthesize scored documents into section drafts with citations.

    On iteration 1: fresh synthesis from sources.
    On iteration 2+: refine previous sections using new sources + gap context.
    """
    research_question = state["research_question"]
    scored_docs = state.get("scored_documents", [])
    current_iteration = state.get("current_iteration", 1)
    prev_sections = state.get("section_drafts", [])
    prev_gaps = state.get("research_gaps", [])
    contradictions = state.get("contradictions", [])

    if not scored_docs:
        return {
            "section_drafts": [
                SectionDraft(
                    id=f"sec-{uuid.uuid4().hex[:8]}",
                    heading="No Results",
                    content="No sources were found for this question.",
                    citation_ids=[],
                    confidence_score=0.0,
                    confidence_level=Confidence.LOW,
                    grader_scores=GraderScores(relevance=0.0, hallucination=1.0, quality=0.0),
                )
            ],
        }

    sources_text, doc_index = _build_source_context(scored_docs)
    top_docs = sorted(scored_docs, key=lambda d: d["combined_score"], reverse=True)[:15]

    # Choose prompt based on iteration
    is_refinement = current_iteration > 1 and prev_sections
    system_prompt = SYNTHESIZER_SYSTEM_REFINE if is_refinement else SYNTHESIZER_SYSTEM_INITIAL

    user_content = f"Research question: {research_question}\n\n"

    if is_refinement:
        prev_context = _build_previous_context(prev_sections, prev_gaps)
        user_content += f"{prev_context}\n\n"

    # Include contradiction context if available
    contradiction_context = _build_contradiction_context(contradictions)
    if contradiction_context:
        user_content += f"{contradiction_context}\n\n"

    user_content += f"Sources ({len(top_docs)} documents):\n{sources_text}"

    data, usage = await caller.call_json(
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        agent_name="synthesizer",
        max_tokens=16384,
    )

    section_drafts, all_citations, research_gaps = _build_citations(data, doc_index, top_docs)

    return {
        "section_drafts": section_drafts,
        "citations": all_citations,
        "research_gaps": research_gaps,
        "token_usage": [usage],
    }

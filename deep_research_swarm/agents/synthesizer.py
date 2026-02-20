"""Synthesizer agent — RAG-Fusion synthesis with inline citations."""

from __future__ import annotations

import json
import uuid

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    GraderScores,
    ScoredDocument,
    SectionDraft,
    SourceAuthority,
)
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.scoring.confidence import classify_confidence

SYNTHESIZER_SYSTEM = """\
You are a research synthesizer. Given a research question and a set of scored source \
documents, produce a structured synthesis with inline citations.

Your job:
1. Identify the key themes and organize them into sections.
2. For each section, synthesize information from the provided sources.
3. Use inline citations in [N] format (e.g., [1], [2]) referencing source numbers.
4. Assess your confidence in each section (0.0 to 1.0).

Output STRICT JSON (no markdown, no commentary):
{
  "sections": [
    {
      "heading": "Section Title",
      "content": "Synthesized content with [1] inline citations [2].",
      "source_ids": ["sd-xxx", "sd-yyy"],
      "confidence": 0.85
    }
  ],
  "gaps": [
    {
      "description": "What information is missing",
      "reason": "no_sources"
    }
  ]
}

Rules:
- Every factual claim MUST have a citation.
- Confidence reflects source quality, agreement, and coverage.
- Gaps: report what you couldn't find or where sources contradict.
- Reasons: "no_sources", "contradictory", "low_confidence"
"""


async def synthesize(state: ResearchState, caller: AgentCaller) -> dict:
    """Synthesize scored documents into section drafts with citations."""
    research_question = state["research_question"]
    scored_docs = state.get("scored_documents", [])

    if not scored_docs:
        return {
            "section_drafts": [
                SectionDraft(
                    id=f"sec-{uuid.uuid4().hex[:8]}",
                    heading="No Results",
                    content="No sources were found for this research question.",
                    citation_ids=[],
                    confidence_score=0.0,
                    confidence_level=Confidence.LOW,
                    grader_scores=GraderScores(
                        relevance=0.0, hallucination=1.0, quality=0.0
                    ),
                )
            ],
        }

    # Build source context for the LLM (top 15 by combined score)
    top_docs = sorted(scored_docs, key=lambda d: d["combined_score"], reverse=True)[:15]
    sources_text = ""
    doc_index: dict[int, ScoredDocument] = {}
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

    user_content = (
        f"Research question: {research_question}\n\n"
        f"Sources ({len(top_docs)} documents):\n{sources_text}"
    )

    data, usage = await caller.call_json(
        system=SYNTHESIZER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="synthesizer",
        max_tokens=4096,
    )

    # Build section drafts
    section_drafts: list[SectionDraft] = []
    all_citations: list[Citation] = []
    citation_counter = 0

    for sec in data.get("sections", []):
        conf_score = sec.get("confidence", 0.5)
        conf_level = classify_confidence(conf_score)

        # Map source references to citations
        source_ids = sec.get("source_ids", [])
        section_citation_ids: list[str] = []

        for src_num_or_id in source_ids:
            citation_counter += 1
            cit_id = f"[{citation_counter}]"

            # Try to find the matching doc
            doc = None
            if isinstance(src_num_or_id, int) and src_num_or_id in doc_index:
                doc = doc_index[src_num_or_id]
            elif isinstance(src_num_or_id, str):
                for d in top_docs:
                    if d["id"] == src_num_or_id:
                        doc = d
                        break

            if doc:
                all_citations.append(
                    Citation(
                        id=cit_id,
                        url=doc["url"],
                        title=doc["title"],
                        authority=doc["authority"],
                        accessed=doc.get("timestamp", ""),
                        used_in_sections=[sec["heading"]],
                    )
                )
                section_citation_ids.append(cit_id)

        sec_id = f"sec-{uuid.uuid4().hex[:8]}"
        section_drafts.append(
            SectionDraft(
                id=sec_id,
                heading=sec["heading"],
                content=sec["content"],
                citation_ids=section_citation_ids,
                confidence_score=round(conf_score, 4),
                confidence_level=conf_level,
                grader_scores=GraderScores(
                    relevance=conf_score,
                    hallucination=1.0,  # Assume no hallucination pre-critique
                    quality=conf_score,
                ),
            )
        )

    # Build research gaps
    from deep_research_swarm.contracts import ResearchGap

    research_gaps: list[ResearchGap] = []
    for gap in data.get("gaps", []):
        research_gaps.append(
            ResearchGap(
                description=gap.get("description", ""),
                attempted_queries=[],
                reason=gap.get("reason", "no_sources"),
            )
        )

    return {
        "section_drafts": section_drafts,
        "citations": all_citations,
        "research_gaps": research_gaps,
        "token_usage": [usage],
    }

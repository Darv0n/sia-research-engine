"""Synthesizer agent — outline-first, mechanically-grounded synthesis (V7, PR-11).

5-stage sub-pipeline:
  Stage 0: Outline validation (deterministic, no LLM)
  Stage 1: Outline generation (1 LLM call)
  Stage 2: Per-section drafting (N parallel LLM calls)
  Stage 3: Mechanical grounding verification (0 LLM calls)
  Stage 4: Refinement of failed sections (M LLM calls, max 2 per section)
  Stage 5: Composition — intro/transitions/conclusion (1 LLM call)

External interface preserved: synthesize(state, caller) -> dict
"""

from __future__ import annotations

import asyncio
import re
import uuid

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    GraderScores,
    ResearchGap,
    ScoredDocument,
    SectionDraft,
    SectionOutline,
    SourcePassage,
)
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.scoring.claim_graph import (
    extract_claims_from_section,
    populate_claim_ids,
)
from deep_research_swarm.scoring.confidence import classify_confidence
from deep_research_swarm.scoring.grounding import (
    assign_passages_to_sections,
    compute_section_grounding_score,
)

# --- Grounding thresholds ---
GROUNDING_PASS_THRESHOLD = 0.8  # Section passes if >= 80% claims grounded
MAX_REFINEMENT_ATTEMPTS = 2

# --- Prompts ---

OUTLINE_SYSTEM = """\
You are a research outline architect. Given a research question, source documents, \
detected contradictions, and source passage statistics, produce a structured outline.

Your job:
1. Identify key themes and organize into {min_sections}-{max_sections} section headings.
   (Do not exceed {max_sections} sections — available source material constrains this.)
2. For each section, list 2-5 specific, concrete claims you intend to make.
3. For each section, identify which source numbers (1-N) are most relevant.
4. Order sections for narrative coherence.
5. If contradictions are present, plan a section or subsection that acknowledges them.

Output STRICT JSON (no markdown, no commentary):
{{
  "sections": [
    {{
      "heading": "Section Title",
      "key_claims": ["Specific claim 1", "Specific claim 2"],
      "source_ids": [1, 3, 7],
      "narrative_role": "introduces the core concept"
    }}
  ],
  "narrative_arc": "Brief description of how sections flow together"
}}

Rules:
- Every claim must be supportable by the listed sources.
- No claim without a source assignment.
- If total passages < 30, keep claims proportionally modest.
- Acknowledge known contradictions — do not paper over them.
"""

SECTION_DRAFT_SYSTEM = """\
You are a research section writer. Write ONE section of a research report.

You will be given:
- The section heading and its narrative role
- Source passages (NOT full documents) to write from
- Specific claims you should support

Rules:
- ONLY use information from the provided passages.
- Every factual claim MUST have an inline [N] citation (N = passage number in input).
- If a passage doesn't support a claim, do NOT cite it for that claim.
- If you cannot support a claim from given passages: write "[insufficient evidence]".
  Do NOT fabricate or extrapolate.
- Write in clear professional prose. Target 400-800 words per section.
- Include analysis and interpretation, not just summary. Explain WHY findings matter.
- Compare and contrast different sources where they offer different perspectives.
- Include specific data points, statistics, or examples from the passages.

Output STRICT JSON (no markdown, no commentary):
{{
  "heading": "Section Title",
  "content": "Written section with [N] citations...",
  "passage_ids_used": ["sp-abc123", "sp-def456"],
  "unsupported_claims": ["Any claims you couldn't ground"]
}}
"""

SECTION_REFINE_SYSTEM = """\
You are refining a research section that has grounding issues.

The following claims did not match their cited passages:
{ungrounded_claims}

For each ungrounded claim, you must:
1. Rewrite the claim to accurately reflect what the source passage says, OR
2. Cite a different passage that does support the original claim, OR
3. Remove the claim and write "[evidence gap]"

Do NOT invent information. If passages don't support a claim, acknowledge it.

Output STRICT JSON (no markdown, no commentary):
{{
  "heading": "Section Title",
  "content": "Revised section with [N] citations...",
  "passage_ids_used": ["sp-abc123"],
  "unsupported_claims": []
}}
"""

COMPOSE_SYSTEM = """\
You are composing a complete research report from pre-verified sections.

Your job:
1. Write a 3-5 sentence introduction that frames the research question, explains why \
it matters, and previews the key findings.
2. Add a 2-3 sentence transition between each pair of sections that explains how \
the topics connect.
3. Write a 4-6 sentence conclusion that synthesizes key findings across ALL sections, \
identifies the strongest evidence, and notes remaining open questions.
4. Do NOT modify section content. Do NOT add new claims or citations.

The sections have been individually verified for source grounding.
Add only narrative connective tissue.

Output STRICT JSON (no markdown, no commentary):
{{
  "introduction": "Opening paragraph...",
  "section_transitions": {{"Section 2 Heading": "Transition text...", ...}},
  "conclusion": "Closing paragraph..."
}}
"""


# --- Helper functions ---


def _build_source_context(
    scored_docs: list[ScoredDocument],
    *,
    max_docs: int = 20,
) -> tuple[str, dict[int, ScoredDocument]]:
    """Build numbered source context for the outline prompt."""
    top_docs = sorted(scored_docs, key=lambda d: d["combined_score"], reverse=True)[:max_docs]
    doc_index: dict[int, ScoredDocument] = {}
    lines: list[str] = []
    for i, doc in enumerate(top_docs, start=1):
        doc_index[i] = doc
        preview = doc["content"][:1000]
        lines.append(
            f"--- Source [{i}] ---\nTitle: {doc['title']}\nURL: {doc['url']}\nContent:\n{preview}"
        )
    return "\n\n".join(lines), doc_index


def _summarize_contradictions(contradictions: list) -> str:
    """Build a contradiction summary for the outline prompt."""
    if not contradictions:
        return ""
    parts = ["Detected contradictions:"]
    for i, c in enumerate(contradictions, 1):
        parts.append(f"{i}. [{c['severity']}] {c['topic']}: {c['claim_a']} vs {c['claim_b']}")
    return "\n".join(parts)


def _build_passage_context(
    passages: list[SourcePassage],
) -> str:
    """Build numbered passage context for section drafting."""
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (id: {p['id']})\n{p['content']}")
    return "\n\n".join(lines)


def _parse_outline_sections(data: dict) -> list[SectionOutline]:
    """Parse LLM outline output into SectionOutline list."""
    sections: list[SectionOutline] = []
    for s in data.get("sections", []):
        source_ids = s.get("source_ids", [])
        # Convert int source_ids to string doc IDs for passage matching
        str_ids = [str(sid) for sid in source_ids]
        sections.append(
            SectionOutline(
                heading=s.get("heading", "Untitled"),
                key_claims=s.get("key_claims", []),
                source_ids=str_ids,
                narrative_role=s.get("narrative_role", ""),
            )
        )
    return sections


# --- Stage 0: Outline Validation ---


def _validate_outline(
    sections: list[SectionOutline],
    passages: list[SourcePassage],
    scored_documents: list[ScoredDocument],
    *,
    max_sections: int = 7,
    min_sections: int = 3,
    max_passages_per_section: int = 8,
) -> tuple[bool, list[str]]:
    """Validate outline structure. Returns (is_valid, failures)."""
    failures: list[str] = []

    # Check 1: source_ids reference existing sources
    available_ids = {str(i) for i in range(1, len(scored_documents) + 1)}
    for section in sections:
        bad_ids = [sid for sid in section.get("source_ids", []) if sid not in available_ids]
        if bad_ids:
            failures.append(
                f"Section '{section['heading']}' references non-existent source IDs: {bad_ids}"
            )

    # Check 2: passage coverage — each section can get passages
    section_passages = assign_passages_to_sections(
        sections, passages, max_passages_per_section=max_passages_per_section
    )
    for section in sections:
        if not section_passages.get(section["heading"]):
            failures.append(f"Section '{section['heading']}' has no assignable passages")

    # Check 3: section count within budget
    if passages:
        budget_max = min(max_sections, max(min_sections, len(passages) // 5))
    else:
        budget_max = min_sections
    if len(sections) > budget_max:
        failures.append(
            f"Outline has {len(sections)} sections but passage budget supports max {budget_max}"
        )

    # Check 4: all sections have key_claims
    for section in sections:
        if not section.get("key_claims"):
            failures.append(f"Section '{section['heading']}' has no key_claims")

    return (len(failures) == 0, failures)


# --- Stage 1: Outline Generation ---


async def _generate_outline(
    state: ResearchState,
    caller: AgentCaller,
    *,
    revision_failures: list[str] | None = None,
    max_sections: int = 7,
    min_sections: int = 3,
    max_docs_for_outline: int = 20,
) -> tuple[list[SectionOutline], dict, list]:
    """Generate outline via LLM. Returns (sections, raw_data, token_usage_list)."""
    scored_docs = state.get("scored_documents", [])
    passages = state.get("source_passages", [])
    contradictions = state.get("contradictions", [])

    total_passage_count = len(passages)
    if total_passage_count:
        effective_max = min(max_sections, max(min_sections, total_passage_count // 5))
    else:
        effective_max = min_sections
    effective_min = min(min_sections, effective_max)

    sources_text, _ = _build_source_context(scored_docs, max_docs=max_docs_for_outline)
    contradiction_summary = _summarize_contradictions(contradictions)

    system = OUTLINE_SYSTEM.format(
        min_sections=effective_min,
        max_sections=effective_max,
    )

    user_parts = [
        f"Research question: {state['research_question']}",
        f"\nTotal source passages available: {total_passage_count}",
        f"\nSources:\n{sources_text}",
    ]

    if contradiction_summary:
        user_parts.append(f"\n{contradiction_summary}")

    if revision_failures:
        user_parts.append(
            "\n\nPREVIOUS OUTLINE WAS INVALID. Fix these issues:\n"
            + "\n".join(f"- {f}" for f in revision_failures)
        )

    data, usage = await caller.call_json(
        system=system,
        messages=[{"role": "user", "content": "\n".join(user_parts)}],
        agent_name="synthesizer_outline",
        max_tokens=4096,
    )

    sections = _parse_outline_sections(data)
    return sections, data, [usage]


# --- Stage 2: Per-Section Drafting ---


async def _draft_section(
    section: SectionOutline,
    passages: list[SourcePassage],
    caller: AgentCaller,
) -> tuple[dict, list]:
    """Draft a single section using assigned passages. Returns (draft_data, usage_list)."""
    if not passages:
        return {
            "heading": section["heading"],
            "content": "[insufficient evidence]",
            "passage_ids_used": [],
            "unsupported_claims": section.get("key_claims", []),
        }, []

    passage_context = _build_passage_context(passages)

    user_content = (
        f"Section heading: {section['heading']}\n"
        f"Narrative role: {section.get('narrative_role', 'evidence')}\n"
        f"Claims to support:\n"
        + "\n".join(f"- {c}" for c in section.get("key_claims", []))
        + f"\n\nSource passages ({len(passages)} available):\n{passage_context}"
    )

    data, usage = await caller.call_json(
        system=SECTION_DRAFT_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="synthesizer_section",
        max_tokens=4096,
    )

    # Ensure heading is correct (LLM may change it)
    data["heading"] = section["heading"]
    return data, [usage]


# --- Citation Renumbering ---


def _build_global_citation_map(
    drafts: list[dict],
    section_passages: dict[str, list[SourcePassage]],
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Build citation mapping: local [N] -> global [M].

    Returns:
      citation_map: {heading: {local_[N]: global_[M]}}
      citation_to_passage_map: {global_[M]: [passage_ids]}
    """
    global_counter = 1
    citation_map: dict[str, dict[str, str]] = {}
    citation_to_passage_map: dict[str, list[str]] = {}

    for draft in drafts:
        heading = draft["heading"]
        passages = section_passages.get(heading, [])
        section_map: dict[str, str] = {}

        for local_n, passage in enumerate(passages, start=1):
            local_key = f"[{local_n}]"
            global_key = f"[{global_counter}]"
            section_map[local_key] = global_key
            citation_to_passage_map[global_key] = [passage["id"]]
            global_counter += 1

        citation_map[heading] = section_map

    return citation_map, citation_to_passage_map


def _renumber_section_content(content: str, section_map: dict[str, str]) -> str:
    """Replace local [N] with global [M] using null-byte placeholder pattern.

    Two-phase replacement prevents double-replacement when numbers overlap
    (e.g. [1] -> [5] and [5] -> [12] would corrupt without placeholders).
    """
    if not section_map:
        return content

    result = content
    placeholders: dict[str, str] = {}

    # Phase 1: replace all local [N] with null-byte placeholders
    for local_key, global_key in section_map.items():
        placeholder = f"\x00{local_key}\x00"
        result = result.replace(local_key, placeholder)
        placeholders[placeholder] = global_key

    # Phase 2: replace placeholders with global numbers
    for placeholder, global_key in placeholders.items():
        result = result.replace(placeholder, global_key)

    return result


# --- Stage 4: Refinement ---


async def _refine_section(
    draft: dict,
    claim_details: list[dict],
    passages: list[SourcePassage],
    caller: AgentCaller,
) -> tuple[dict, list]:
    """Refine a section with grounding issues. Returns (refined_data, usage_list)."""
    ungrounded = [d for d in claim_details if not d.get("grounded")]
    if not ungrounded:
        return draft, []

    claims_formatted = "\n".join(
        f'- Claim: "{d["claim"]}" (cited {d["citation_id"]}, similarity={d["similarity"]:.2f})'
        for d in ungrounded
    )

    passage_context = _build_passage_context(passages)
    system = SECTION_REFINE_SYSTEM.format(ungrounded_claims=claims_formatted)

    user_content = (
        f"Section: {draft['heading']}\n"
        f"Current content:\n{draft['content']}\n\n"
        f"Available passages:\n{passage_context}"
    )

    data, usage = await caller.call_json(
        system=system,
        messages=[{"role": "user", "content": user_content}],
        agent_name="synthesizer_refine",
        max_tokens=4096,
    )

    data["heading"] = draft["heading"]
    return data, [usage]


# --- Stage 5: Composition ---


async def _compose_report(
    sections: list[dict],
    research_question: str,
    caller: AgentCaller,
) -> tuple[dict, list]:
    """Generate intro, transitions, conclusion. Returns (composition, usage_list)."""
    section_summary = "\n".join(
        f"- {s['heading']}: {s.get('content', '')[:200]}..." for s in sections
    )

    user_content = f"Research question: {research_question}\n\nSections:\n{section_summary}"

    data, usage = await caller.call_json(
        system=COMPOSE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        agent_name="synthesizer_compose",
        max_tokens=4096,
    )

    return data, [usage]


# --- Output Builder ---


def _build_output(
    verified_sections: list[tuple[dict, float, list[dict]]],
    composition: dict,
    citation_map: dict[str, dict[str, str]],
    citation_to_passage_map: dict[str, list[str]],
    section_passages: dict[str, list[SourcePassage]],
    research_gaps: list[ResearchGap],
) -> dict:
    """Build the synthesize() return dict preserving grounding data."""
    section_drafts: list[SectionDraft] = []
    all_citations: list[Citation] = []
    citation_re = re.compile(r"\[(\d+)\]")

    for draft, grounding_score, claim_details in verified_sections:
        heading = draft["heading"]
        content = draft.get("content", "")

        # Renumber citations local -> global
        section_map = citation_map.get(heading, {})
        renumbered = _renumber_section_content(content, section_map)

        # Extract citation IDs from renumbered content
        refs = citation_re.findall(renumbered)
        cit_ids = list(dict.fromkeys(f"[{r}]" for r in refs))

        # LLM-provided confidence (or default)
        conf_score = draft.get("confidence", 0.7)
        conf_level = classify_confidence(conf_score)

        section_drafts.append(
            SectionDraft(
                id=f"sec-{uuid.uuid4().hex[:8]}",
                heading=heading,
                content=renumbered,
                citation_ids=cit_ids,
                confidence_score=round(conf_score, 4),
                confidence_level=conf_level,
                grader_scores=GraderScores(
                    relevance=conf_score,
                    hallucination=1.0 - (1.0 - grounding_score) * 0.5,
                    quality=conf_score,
                ),
                grounding_score=grounding_score,
                claim_details=claim_details,
            )
        )

        # Build Citation objects from renumbered refs
        passages = section_passages.get(heading, [])
        for cid in cit_ids:
            # Find passage for this citation
            passage_ids = citation_to_passage_map.get(cid, [])
            if passage_ids:
                # Find the passage to get URL info
                for p in passages:
                    if p["id"] in passage_ids:
                        all_citations.append(
                            Citation(
                                id=cid,
                                url=p.get("source_id", ""),
                                title=heading,
                                authority="unknown",
                                accessed="",
                                used_in_sections=[heading],
                            )
                        )
                        break

    return {
        "section_drafts": section_drafts,
        "citations": all_citations,
        "citation_to_passage_map": citation_to_passage_map,
        "research_gaps": research_gaps,
        "composition": composition,
    }


# --- Main entry point ---


async def synthesize(state: ResearchState, caller: AgentCaller) -> dict:
    """Synthesize scored documents into section drafts with citations.

    5-stage outline-first pipeline with mechanical grounding verification.
    External interface preserved from V6.
    """
    scored_docs = state.get("scored_documents", [])
    passages = state.get("source_passages", [])
    current_iteration = state.get("current_iteration", 1)
    prev_sections = state.get("section_drafts", [])

    # Read adaptive tunables (V8) — fall back to V7 hardcoded defaults
    _snap = state.get("tunable_snapshot", {})
    grounding_pass = _snap.get("grounding_pass_threshold", GROUNDING_PASS_THRESHOLD)
    max_refine = int(_snap.get("max_refinement_attempts", MAX_REFINEMENT_ATTEMPTS))
    max_secs = int(_snap.get("max_sections", 7))
    min_secs = int(_snap.get("min_sections", 3))
    max_docs_outline = int(_snap.get("max_docs_for_outline", 20))
    max_pp_sec = int(_snap.get("max_passages_per_section", 8))

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
                    grader_scores=GraderScores(
                        relevance=0.0,
                        hallucination=1.0,
                        quality=0.0,
                    ),
                )
            ],
        }

    all_usage: list = []

    # --- Stage 1: Generate outline ---
    sections, outline_data, usage = await _generate_outline(
        state,
        caller,
        max_sections=max_secs,
        min_sections=min_secs,
        max_docs_for_outline=max_docs_outline,
    )
    all_usage.extend(usage)

    # --- Stage 0: Validate outline ---
    is_valid, failures = _validate_outline(
        sections,
        passages,
        scored_docs,
        max_sections=max_secs,
        min_sections=min_secs,
        max_passages_per_section=max_pp_sec,
    )

    if not is_valid:
        # One revision pass
        sections, outline_data, usage = await _generate_outline(
            state,
            caller,
            revision_failures=failures,
            max_sections=max_secs,
            min_sections=min_secs,
            max_docs_for_outline=max_docs_outline,
        )
        all_usage.extend(usage)

        is_valid, failures = _validate_outline(
            sections,
            passages,
            scored_docs,
            max_sections=max_secs,
            min_sections=min_secs,
            max_passages_per_section=max_pp_sec,
        )
        if not is_valid:
            # Drop sections with no assignable passages
            section_passages_check = assign_passages_to_sections(
                sections,
                passages,
                max_passages_per_section=max_pp_sec,
            )
            sections = [s for s in sections if section_passages_check.get(s["heading"])]

    if not sections:
        return {
            "section_drafts": [
                SectionDraft(
                    id=f"sec-{uuid.uuid4().hex[:8]}",
                    heading="Insufficient Coverage",
                    content="Source material was insufficient to produce a structured report.",
                    citation_ids=[],
                    confidence_score=0.2,
                    confidence_level=Confidence.LOW,
                    grader_scores=GraderScores(
                        relevance=0.2,
                        hallucination=1.0,
                        quality=0.2,
                    ),
                )
            ],
            "token_usage": all_usage,
        }

    # --- Assign passages to sections ---
    section_passages = assign_passages_to_sections(
        sections,
        passages,
        max_passages_per_section=max_pp_sec,
    )

    # Iteration 2+: keep HIGH grounding sections from previous iteration
    keep_headings: set[str] = set()
    kept_drafts: dict[str, dict] = {}
    if current_iteration > 1 and prev_sections:
        for prev in prev_sections:
            gs = prev.get("grounding_score", 0)
            if gs >= grounding_pass:
                conf = prev.get("confidence_level")
                if conf == Confidence.HIGH or conf == "HIGH":
                    keep_headings.add(prev["heading"])
                    kept_drafts[prev["heading"]] = {
                        "heading": prev["heading"],
                        "content": prev["content"],
                        "passage_ids_used": [],
                        "unsupported_claims": [],
                        "confidence": prev.get("confidence_score", 0.8),
                    }

    # --- Stage 2: Draft sections in parallel ---
    draft_tasks = []
    draft_order: list[str] = []  # Track which headings were drafted

    for section in sections:
        heading = section["heading"]
        if heading in keep_headings:
            draft_order.append(heading)
            continue
        draft_order.append(heading)
        section_p = section_passages.get(heading, [])
        draft_tasks.append(_draft_section(section, section_p, caller))

    draft_results = await asyncio.gather(*draft_tasks)

    # Merge kept drafts with new drafts
    all_drafts: list[dict] = []
    draft_idx = 0
    for heading in draft_order:
        if heading in kept_drafts:
            all_drafts.append(kept_drafts[heading])
        else:
            draft_data, usage = draft_results[draft_idx]
            all_usage.extend(usage)
            all_drafts.append(draft_data)
            draft_idx += 1

    # --- Build citation maps ---
    citation_map, citation_to_passage_map = _build_global_citation_map(
        all_drafts,
        section_passages,
    )

    # --- Stage 3: Mechanical grounding verification ---
    verified: list[tuple[dict, float, list[dict]]] = []
    needs_refinement: list[tuple[dict, float, list[dict], str]] = []

    for draft in all_drafts:
        heading = draft["heading"]
        content = draft.get("content", "")

        # Get renumbered citation IDs for this section
        section_map = citation_map.get(heading, {})
        renumbered = _renumber_section_content(content, section_map)
        cit_ids = list(dict.fromkeys(f"[{r}]" for r in re.findall(r"\[(\d+)\]", renumbered)))

        score, details = compute_section_grounding_score(
            renumbered,
            cit_ids,
            passages,
            citation_to_passage_map,
        )

        if score >= grounding_pass or heading in keep_headings:
            verified.append((draft, score, details))
        else:
            needs_refinement.append((draft, score, details, heading))

    # --- Stage 4: Refine sections below threshold ---
    for draft, score, details, heading in needs_refinement:
        refined_draft = draft
        refined_score = score
        refined_details = details

        for _attempt in range(max_refine):
            section_p = section_passages.get(heading, [])
            refined_draft, usage = await _refine_section(
                refined_draft,
                refined_details,
                section_p,
                caller,
            )
            all_usage.extend(usage)

            # Re-verify
            section_map = citation_map.get(heading, {})
            renumbered = _renumber_section_content(
                refined_draft.get("content", ""),
                section_map,
            )
            cit_ids = list(dict.fromkeys(f"[{r}]" for r in re.findall(r"\[(\d+)\]", renumbered)))

            refined_score, refined_details = compute_section_grounding_score(
                renumbered,
                cit_ids,
                passages,
                citation_to_passage_map,
            )

            if refined_score >= grounding_pass:
                break

        # Accept section regardless (critique handles remaining issues)
        verified.append((refined_draft, refined_score, refined_details))

    # --- Stage 5: Composition ---
    composition, usage = await _compose_report(
        [v[0] for v in verified],
        state["research_question"],
        caller,
    )
    all_usage.extend(usage)

    # --- Build research gaps from unsupported claims ---
    research_gaps: list[ResearchGap] = []
    for draft, _, _ in verified:
        for claim in draft.get("unsupported_claims", []):
            if claim:
                research_gaps.append(
                    ResearchGap(
                        description=claim,
                        attempted_queries=[],
                        reason="no_sources",
                    )
                )

    # --- Build output ---
    result = _build_output(
        verified,
        composition,
        citation_map,
        citation_to_passage_map,
        section_passages,
        research_gaps,
    )
    result["token_usage"] = all_usage

    # --- Populate claim graph (V8, OE1) ---
    all_claims: list[dict] = []
    for sec in result.get("section_drafts", []):
        all_claims.extend(extract_claims_from_section(sec))
    if all_claims:
        updated_passages = populate_claim_ids(
            passages, all_claims, result.get("citation_to_passage_map", {})
        )
        result["source_passages"] = updated_passages

    return result

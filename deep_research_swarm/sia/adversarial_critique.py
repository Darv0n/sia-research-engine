"""Adversarial critique — multi-turn evaluation using SIA cognitive lenses.

Replaces V9's 3 parallel graders with a conversation where agents challenge
the synthesis from different cognitive stances. Each agent sees prior agents'
critiques and builds on them.

Sequence:
  1. Makishima — legitimacy stress-test, ethical faultline exposure
  2. Lawliet  — constraint extraction, grounding validation
  3. Rick     — frame audit (conditional on entropy > 0.40)
  4. Shikamaru — synthesis readiness, pruning
  5. Light    — replan direction (conditional on replan recommendation)

Score adjustment per finding severity:
  critical:    -0.15
  significant: -0.08
  minor:       -0.03
"""

from __future__ import annotations

import json
import re
from typing import Any

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.contracts import (
    AdversarialFinding,
    CritiqueTrace,
    SectionDraft,
)

# Score adjustments by severity
SEVERITY_ADJUSTMENTS: dict[str, float] = {
    "critical": -0.15,
    "significant": -0.08,
    "minor": -0.03,
}

# Adversarial critique sequence (agent_id, required flag)
# required=False means conditional activation
CRITIQUE_SEQUENCE: list[tuple[str, bool]] = [
    ("makishima", True),
    ("lawliet", True),
    ("rick", False),  # conditional: entropy > 0.40
    ("shikamaru", True),
    ("light", False),  # conditional: replan recommended
]

ADVERSARIAL_SYSTEM = """\
You are conducting an adversarial critique of a research synthesis.

Your cognitive stance: {agent_name} ({agent_archetype})
Your processing bias: {processing_bias}

Research question: {research_question}

Prior critique conversation:
{prior_critiques}

Sections to evaluate:
{sections_text}

For EACH finding, assess:
- target_section: which section_id (or "global" for cross-cutting issues)
- finding: specific, actionable description of the issue
- severity: "critical" (fundamental flaw) | "significant" (weakens argument) | \
"minor" (could be improved)
- actionable: true if the finding can be addressed with available evidence

Respond with STRICT JSON:
{{
  "findings": [
    {{
      "target_section": "sec-xxx",
      "finding": "description",
      "severity": "critical|significant|minor",
      "actionable": true
    }}
  ],
  "constraints": ["any new constraints extracted"],
  "recommendation": "converge|replan|refine_targeted",
  "missing_variables": ["variables that remain unresolved"],
  "alternative_frames": ["alternative interpretive frames not yet considered"]
}}
"""

# Per-agent processing bias descriptions
PROCESSING_BIASES: dict[str, str] = {
    "makishima": (
        "TEST LEGITIMACY — Identify value assumptions treated as facts. "
        "Stress-test institutional authority filling evidence gaps. "
        "Find whose perspective is missing."
    ),
    "lawliet": (
        "EXTRACT CONSTRAINTS — Identify unsupported claims. "
        "Check grounding: are citations actually supporting the claims made? "
        "Extract what MUST be true vs what is merely asserted."
    ),
    "rick": (
        "FRAME AUDIT — Identify the frame everyone is operating within. "
        "What hidden assumptions drive the synthesis? "
        "What would change if the opposite of the consensus were true?"
    ),
    "shikamaru": (
        "READINESS CHECK — Is the synthesis sufficient? "
        "Identify redundant sections, ornamental arguments, and prunable branches. "
        "Assess: can we ship this, or does it need more work?"
    ),
    "light": (
        "REPLAN DIRECTION — Given the critique findings, what should change? "
        "If replanning: which specific queries would fill the gaps? "
        "If converging: confirm the directional commitment is sound."
    ),
}


def _should_activate_agent(
    agent_id: str,
    entropy_value: float,
    has_replan_recommendation: bool,
) -> bool:
    """Check if a conditional agent should be activated."""
    if agent_id == "rick":
        return entropy_value > 0.40
    if agent_id == "light":
        return has_replan_recommendation
    return True


def _parse_critique_output(
    agent_id: str,
    raw_output: str,
) -> dict[str, Any]:
    """Parse agent critique output, with fallback for non-JSON responses."""
    # Try JSON parse
    try:
        data = json.loads(raw_output)
        return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON block
    json_match = re.search(r"\{[\s\S]*\}", raw_output)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract what we can from text
    findings: list[dict[str, Any]] = []

    # Look for severity keywords
    for severity in ("critical", "significant", "minor"):
        pattern = rf"(?:^|\n)\s*[-*•]\s*.*{severity}.*?[:]\s*(.*?)(?:\n|$)"
        for match in re.finditer(pattern, raw_output, re.IGNORECASE):
            findings.append(
                {
                    "target_section": "global",
                    "finding": match.group(1).strip(),
                    "severity": severity,
                    "actionable": True,
                }
            )

    # If no findings extracted, create a single global finding from the text
    if not findings and raw_output.strip():
        findings.append(
            {
                "target_section": "global",
                "finding": raw_output.strip()[:500],
                "severity": "minor",
                "actionable": True,
            }
        )

    return {
        "findings": findings,
        "constraints": [],
        "recommendation": "converge",
        "missing_variables": [],
        "alternative_frames": [],
    }


def _build_sections_text(section_drafts: list[SectionDraft]) -> str:
    """Build section text for critique evaluation."""
    parts: list[str] = []
    for sec in section_drafts:
        parts.append(f"--- Section: {sec['heading']} (id: {sec['id']}) ---\n{sec['content']}")
    return "\n\n".join(parts)


def _apply_score_adjustments(
    section_drafts: list[SectionDraft],
    findings: list[AdversarialFinding],
) -> list[SectionDraft]:
    """Apply score adjustments from adversarial findings to section drafts.

    Returns updated section drafts with adjusted confidence scores.
    """
    from deep_research_swarm.scoring.confidence import classify_confidence

    # Group findings by section
    section_adjustments: dict[str, float] = {}
    for finding in findings:
        target = finding["target_section"]
        adj = SEVERITY_ADJUSTMENTS.get(finding["severity"], 0.0)

        if target == "global":
            # Global findings apply to all sections (reduced impact)
            for sec in section_drafts:
                section_adjustments[sec["id"]] = section_adjustments.get(sec["id"], 0.0) + adj * 0.5
        else:
            section_adjustments[target] = section_adjustments.get(target, 0.0) + adj

    updated: list[SectionDraft] = []
    for sec in section_drafts:
        adj = section_adjustments.get(sec["id"], 0.0)
        new_score = max(0.0, min(1.0, sec["confidence_score"] + adj))

        # Copy all existing fields (including NotRequired like grounding_score,
        # claim_details) then override only confidence fields
        adjusted = dict(sec)
        adjusted["confidence_score"] = round(new_score, 4)
        adjusted["confidence_level"] = classify_confidence(new_score)
        updated.append(adjusted)  # type: ignore[arg-type]

    return updated


async def adversarial_critique(
    state: dict[str, Any],
    caller: AgentCaller,
    *,
    convergence_threshold: float = 0.05,
) -> dict:
    """Run multi-turn adversarial critique using SIA cognitive lenses.

    Returns a dict compatible with the existing critique consumer contract:
    section_drafts, converged, convergence_reason, iteration_history, token_usage,
    plus adversarial_findings and critique_trace.
    """
    from deep_research_swarm.contracts import (
        IterationRecord,
        SectionConfidenceSnapshot,
    )
    from deep_research_swarm.scoring.confidence import (
        should_replan,
        summarize_confidence,
    )
    from deep_research_swarm.sia.agents import AGENT_BY_ID

    research_question = state.get("research_question", "")
    section_drafts: list[SectionDraft] = state.get("section_drafts", [])
    current_iteration = state.get("current_iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    total_tokens = state.get("total_tokens_used", 0)
    token_budget = state.get("token_budget", 200000)

    # Get entropy for conditional activation
    entropy_state = state.get("entropy_state", {})
    entropy_value = entropy_state.get("e", 0.35)

    if not section_drafts:
        return {
            "converged": True,
            "convergence_reason": "no_sections_to_evaluate",
        }

    sections_text = _build_sections_text(section_drafts)

    # Shared conversation thread
    conversation: list[str] = []
    all_findings: list[AdversarialFinding] = []
    all_constraints: list[str] = []
    all_missing_vars: list[str] = []
    all_alt_frames: list[str] = []
    all_usages: list[dict] = []
    has_replan = False
    has_refine_targeted = False
    turns_executed = 0

    for agent_id, required in CRITIQUE_SEQUENCE:
        # Check conditional activation
        if not required and not _should_activate_agent(agent_id, entropy_value, has_replan):
            continue

        agent = AGENT_BY_ID.get(agent_id)
        if agent is None:
            continue

        # Build prior critiques text
        prior_text = "\n\n".join(conversation) if conversation else "(No prior critiques)"

        # Build the system prompt
        system = ADVERSARIAL_SYSTEM.format(
            agent_name=agent.name,
            agent_archetype=agent.archetype,
            processing_bias=PROCESSING_BIASES.get(agent_id, "Evaluate critically."),
            research_question=research_question,
            prior_critiques=prior_text,
            sections_text=sections_text,
        )

        # Call the model
        try:
            data, usage = await caller.call_json(
                system=system,
                messages=[{"role": "user", "content": "Conduct your adversarial critique."}],
                agent_name=f"adversarial_{agent_id}",
                max_tokens=2048,
            )
        except Exception:
            # Parse the error case — skip this agent
            data = {"findings": [], "recommendation": "converge"}
            usage = {}

        all_usages.append(usage)
        turns_executed += 1

        # Parse output
        parsed = data if isinstance(data, dict) else _parse_critique_output(agent_id, str(data))

        # Extract findings
        for f in parsed.get("findings", []):
            finding = AdversarialFinding(
                agent=agent_id,
                int_type=agent.preferred_int_types[0] if agent.preferred_int_types else "C",
                target_section=f.get("target_section", "global"),
                finding=f.get("finding", ""),
                severity=f.get("severity", "minor"),
                actionable=f.get("actionable", True),
                response_to=conversation[-1][:200] if conversation else "",
            )
            all_findings.append(finding)

        # Accumulate
        for c in parsed.get("constraints", []):
            if c and c not in all_constraints:
                all_constraints.append(c)
        for mv in parsed.get("missing_variables", []):
            if mv and mv not in all_missing_vars:
                all_missing_vars.append(mv)
        for af in parsed.get("alternative_frames", []):
            if af and af not in all_alt_frames:
                all_alt_frames.append(af)

        # Check for replan/refine recommendation
        rec = parsed.get("recommendation", "converge")
        if rec == "replan":
            has_replan = True
        elif rec == "refine_targeted":
            has_refine_targeted = True

        # Add to conversation thread
        finding_summary = "; ".join(f.get("finding", "")[:100] for f in parsed.get("findings", []))
        conversation.append(
            f"[{agent.name}] Recommendation: {rec}. "
            f"Findings ({len(parsed.get('findings', []))}): {finding_summary}"
        )

    # Apply score adjustments
    updated_sections = _apply_score_adjustments(section_drafts, all_findings)

    # Determine convergence using same logic as classic critique
    prev_history = state.get("iteration_history", [])
    prev_avg = prev_history[-1]["avg_confidence"] if prev_history else 0.0

    replan, reason = should_replan(
        updated_sections, prev_avg=prev_avg, delta_threshold=convergence_threshold
    )

    # Override: only a full replan recommendation forces re-iteration
    # refine_targeted does NOT trigger replan — it's handled via final_rec
    if has_replan and not replan:
        replan = True
        reason = "adversarial_replan_recommended"

    # Force convergence conditions
    if current_iteration >= max_iterations:
        replan = False
        reason = f"max_iterations_reached ({max_iterations})"

    _snap = state.get("tunable_snapshot", {})
    budget_exhaust_pct = _snap.get("budget_exhaustion_pct", 0.9)
    # Include tokens consumed by the critique itself (not just pre-critique count)
    critique_tokens = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in all_usages)
    effective_total = total_tokens + critique_tokens
    if effective_total > token_budget * budget_exhaust_pct:
        replan = False
        reason = f"budget_nearly_exhausted ({effective_total}/{token_budget})"

    converged = not replan

    # Determine final recommendation
    critical_count = sum(1 for f in all_findings if f["severity"] == "critical")
    if has_replan:
        final_rec = "replan"
    elif critical_count > 0 and converged:
        final_rec = "refine_targeted"
    elif has_refine_targeted:
        final_rec = "refine_targeted"
    else:
        final_rec = "converge"

    # Build critique trace
    trace = CritiqueTrace(
        turns=turns_executed,
        findings_count=len(all_findings),
        critical_findings=critical_count,
        constraints_extracted=len(all_constraints),
        missing_variables=all_missing_vars,
        alternative_frames=all_alt_frames,
        recommendation=final_rec,
    )

    # Build iteration record (same shape as classic critique)
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
        "adversarial_findings": all_findings,
        "critique_trace": dict(trace),
    }

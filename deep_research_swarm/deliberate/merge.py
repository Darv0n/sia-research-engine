"""Judgment reconciliation — merge 4-judge outputs into JudgmentContext.

Deterministic, zero LLM calls. Cross-references:
  authority x grounding  -> ClaimVerdict (full)
  contradiction x coverage -> ActiveTension (annotated)
  diversity x grounding -> structural_risks

Produces the unified JudgmentContext that feeds into compression.
"""

from __future__ import annotations

from typing import Any

from deep_research_swarm.contracts import (
    ActiveTension,
    ClaimVerdict,
    CoverageMap,
    JudgmentContext,
    SubQuery,
)


def merge_judgments(
    authority_output: dict[str, Any],
    grounding_output: dict[str, Any],
    contradiction_output: dict[str, Any],
    coverage_output: dict[str, Any],
    *,
    wave_number: int = 1,
) -> JudgmentContext:
    """Merge 4-judge outputs into a unified JudgmentContext.

    Cross-references authority and grounding verdicts to produce full
    ClaimVerdict objects. Annotates tensions with authority differential.
    Identifies structural risks from coverage gaps + grounding weakness.
    """
    source_credibility = authority_output.get("source_credibility", {})
    authority_verdicts = authority_output.get("claim_verdicts_authority", [])
    grounding_verdicts = grounding_output.get("claim_verdicts_grounding", [])
    tensions = contradiction_output.get("active_tensions", [])
    coverage_map: CoverageMap = coverage_output.get(
        "coverage_map",
        {
            "facet_coverage": {},
            "overall_coverage": 0.0,
            "uncovered_facets": [],
            "under_represented_perspectives": [],
        },
    )
    next_queries = coverage_output.get("next_wave_queries", [])

    # --- Cross-reference: authority x grounding -> full ClaimVerdict ---
    # Build authority lookup by passage_id
    auth_by_passage: dict[str, dict] = {}
    for av in authority_verdicts:
        pid = av.get("passage_id", "")
        if pid:
            auth_by_passage[pid] = av

    # Build grounding -> passage reverse map
    passage_to_claims = grounding_output.get("passage_to_claims", {})
    claim_to_passages: dict[str, list[str]] = {}
    for pid, cids in passage_to_claims.items():
        for cid in cids:
            claim_to_passages.setdefault(cid, []).append(pid)

    # Merge into full ClaimVerdicts
    claim_verdicts: list[ClaimVerdict] = []
    contradicted_claims = _extract_contradicted_claim_ids(tensions)

    for gv in grounding_verdicts:
        claim_id = gv.get("claim_id", "")
        # Find authority data from linked passages
        linked_passages = claim_to_passages.get(claim_id, [])
        auth_score = 0.0
        auth_level = "unknown"
        for pid in linked_passages:
            auth = auth_by_passage.get(pid, {})
            s = auth.get("authority_score", 0.0)
            if s > auth_score:
                auth_score = s
                auth_level = auth.get("authority_level", "unknown")

        is_contradicted = claim_id in contradicted_claims
        contradiction_id = contradicted_claims.get(claim_id) if is_contradicted else None

        verdict = ClaimVerdict(
            claim_id=claim_id,
            claim_text=gv.get("claim_text", ""),
            grounding_score=gv.get("grounding_score", 0.0),
            grounding_method=gv.get("grounding_method", "unverified"),
            authority_score=round(auth_score, 4),
            authority_level=auth_level,
            contradicted=is_contradicted,
        )
        # Add optional contradiction_id if present
        if contradiction_id:
            verdict["contradiction_id"] = contradiction_id

        claim_verdicts.append(verdict)

    # --- Annotate tensions with authority differential ---
    annotated_tensions = _annotate_tensions(tensions, source_credibility)

    # --- Structural risks ---
    structural_risks = _compute_structural_risks(
        claim_verdicts,
        coverage_map,
        grounding_verdicts,
    )

    # --- Overall coverage ---
    overall_coverage = coverage_map.get("overall_coverage", 0.0)

    # --- Next wave queries as SubQuery-typed ---
    typed_queries: list[SubQuery] = []
    for q in next_queries:
        typed_queries.append(
            SubQuery(
                id=q.get("id", ""),
                question=q.get("question", ""),
                backends=q.get("backends", ["searxng"]),
            )
        )

    return JudgmentContext(
        claim_verdicts=claim_verdicts,
        source_credibility=source_credibility,
        active_tensions=annotated_tensions,
        coverage_map=coverage_map,
        next_wave_queries=typed_queries,
        overall_coverage=round(overall_coverage, 4),
        structural_risks=structural_risks,
        wave_number=wave_number,
    )


def _extract_contradicted_claim_ids(
    tensions: list[ActiveTension],
) -> dict[str, str]:
    """Build {claim_id: tension_id} for claims involved in contradictions."""
    result: dict[str, str] = {}
    for t in tensions:
        tid = t.get("id", "")
        claim_a = t.get("claim_a", {})
        claim_b = t.get("claim_b", {})
        if claim_a.get("claim_id"):
            result[claim_a["claim_id"]] = tid
        if claim_b.get("claim_id"):
            result[claim_b["claim_id"]] = tid
    return result


def _annotate_tensions(
    tensions: list[ActiveTension],
    source_credibility: dict[str, float],
) -> list[ActiveTension]:
    """Annotate tensions with authority differential from source credibility.

    If both claims have known source URLs, compute the absolute difference
    in their authority scores. Higher differential = stronger asymmetry.
    """
    # For now, tensions don't carry source URLs, so differential stays at 0.
    # When passages are linked, this can compute real differentials.
    return list(tensions)


def _compute_structural_risks(
    claim_verdicts: list[ClaimVerdict],
    coverage_map: CoverageMap,
    grounding_verdicts: list[dict],
) -> list[str]:
    """Identify structural risks from cross-referenced judgments."""
    risks: list[str] = []

    # Risk: low overall coverage
    if coverage_map.get("overall_coverage", 0) < 0.4:
        risks.append("low_coverage: overall coverage below 40%")

    # Risk: many uncovered facets
    uncovered = coverage_map.get("uncovered_facets", [])
    if len(uncovered) >= 3:
        risks.append(f"uncovered_facets: {len(uncovered)} facets below 30% coverage")

    # Risk: weak grounding
    grounded_count = sum(1 for v in grounding_verdicts if v.get("grounded", False))
    total_claims = max(len(grounding_verdicts), 1)
    grounding_ratio = grounded_count / total_claims
    if grounding_ratio < 0.5 and total_claims >= 3:
        risks.append(f"weak_grounding: only {grounding_ratio:.0%} of claims grounded")

    # Risk: under-represented perspectives
    under_rep = coverage_map.get("under_represented_perspectives", [])
    if under_rep:
        risks.append(f"perspective_gap: missing {', '.join(under_rep)}")

    # Risk: high contradiction density
    contradicted = sum(1 for v in claim_verdicts if v.get("contradicted", False))
    if contradicted >= 3:
        risks.append(f"high_contradiction: {contradicted} claims contradicted")

    return risks

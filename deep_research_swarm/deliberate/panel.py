"""Deliberation panel — 4 parallel judges producing structured verdicts.

Each judge reuses existing scoring/agent modules and returns typed dicts
compatible with merge.py's reconciliation logic.

All judges are designed to run in parallel (no cross-dependencies).
"""

from __future__ import annotations

import hashlib
from typing import Any

from deep_research_swarm.contracts import (
    ActiveTension,
    AuthorityProfile,
    ClaimVerdict,
    CoverageMap,
    Facet,
    ScoredDocument,
    SourcePassage,
)
from deep_research_swarm.scoring.authority import score_authority
from deep_research_swarm.scoring.claim_graph import (
    extract_claims_from_section,
    link_claims_to_passages,
)
from deep_research_swarm.scoring.diversity import compute_diversity
from deep_research_swarm.scoring.grounding import verify_grounding

# ============================================================
# Authority Judge
# ============================================================


def authority_judge(
    scored_documents: list[ScoredDocument],
    source_passages: list[SourcePassage],
) -> dict[str, Any]:
    """Score source credibility across all documents.

    Deterministic — zero LLM calls. Reuses scoring/authority.py.

    Returns:
        {
            "source_credibility": {url: float},
            "authority_profiles": [AuthorityProfile],
            "claim_verdicts_authority": [partial ClaimVerdict with authority fields],
        }
    """
    source_credibility: dict[str, float] = {}
    url_authority_level: dict[str, str] = {}  # cache level per URL
    authority_counts: dict[str, int] = {}

    for doc in scored_documents:
        url = doc.get("url", "")
        if url and url not in source_credibility:
            scholarly = doc.get("scholarly_metadata")
            _auth, score = score_authority(url, scholarly_metadata=scholarly)
            source_credibility[url] = score
            level = _auth.value if hasattr(_auth, "value") else str(_auth)
            url_authority_level[url] = level
            authority_counts[level] = authority_counts.get(level, 0) + 1

    # Build authority profile from aggregate
    total = max(sum(authority_counts.values()), 1)
    scores = list(source_credibility.values())
    avg_score = sum(scores) / max(len(scores), 1)

    # Find dominant authority level
    dominant = max(authority_counts, key=authority_counts.get) if authority_counts else "unknown"
    institutional = authority_counts.get("institutional", 0)

    profile = AuthorityProfile(
        dominant_authority=dominant,
        source_count=total,
        avg_authority_score=round(avg_score, 4),
        institutional_ratio=round(institutional / total, 4),
    )

    # Partial claim verdicts — authority fields only (merged with grounding later)
    partial_verdicts: list[dict] = []
    for passage in source_passages:
        url = passage.get("source_url", "")
        auth_score = source_credibility.get(url, 0.4)
        level = url_authority_level.get(url, "unknown")
        partial_verdicts.append(
            {
                "passage_id": passage.get("id", ""),
                "authority_score": auth_score,
                "authority_level": level,
            }
        )

    return {
        "source_credibility": source_credibility,
        "authority_profiles": [profile],
        "claim_verdicts_authority": partial_verdicts,
    }


# ============================================================
# Grounding Judge
# ============================================================


def grounding_judge(
    section_drafts: list[dict],
    source_passages: list[SourcePassage],
    citation_to_passage_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Verify claim grounding via Jaccard + optional embedding.

    Deterministic — zero LLM calls. Reuses scoring/grounding.py + claim_graph.py.

    Returns:
        {
            "claim_verdicts_grounding": [partial ClaimVerdict with grounding fields],
            "passage_to_claims": {passage_id: [claim_id]},
        }
    """
    # Build passage lookup
    passage_map: dict[str, SourcePassage] = {
        p.get("id", ""): p for p in source_passages if p.get("id")
    }

    verdicts: list[dict] = []
    passage_to_claims: dict[str, list[str]] = {}

    for section in section_drafts:
        claims = extract_claims_from_section(section)
        claim_links = link_claims_to_passages(claims, citation_to_passage_map)

        for claim in claims:
            claim_id = claim["id"]
            linked_passages = claim_links.get(claim_id, [])

            best_score = 0.0
            best_method = "unverified"
            grounded = False

            for pid in linked_passages:
                passage = passage_map.get(pid)
                if not passage:
                    continue

                ok, sim, method = verify_grounding(
                    claim["text"],
                    passage,
                )
                if sim > best_score:
                    best_score = sim
                    best_method = method
                    grounded = ok

                # Track reverse mapping
                passage_to_claims.setdefault(pid, []).append(claim_id)

            verdicts.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim["text"],
                    "grounding_score": round(best_score, 4),
                    "grounding_method": best_method,
                    "grounded": grounded,
                }
            )

    return {
        "claim_verdicts_grounding": verdicts,
        "passage_to_claims": passage_to_claims,
    }


# ============================================================
# Contradiction Judge
# ============================================================


async def contradiction_judge(
    scored_documents: list[ScoredDocument],
    caller: Any,
    *,
    max_docs: int = 10,
) -> dict[str, Any]:
    """Detect contradictions and annotate with authority data.

    1 Sonnet call via agents/contradiction.py core logic.

    Returns:
        {
            "contradictions": [Contradiction],
            "active_tensions": [ActiveTension],
            "token_usage": [usage],
        }
    """
    from deep_research_swarm.agents.contradiction import detect_contradictions

    contradictions, usage = await detect_contradictions(
        scored_documents,
        caller,
        max_docs=max_docs,
    )

    # Promote contradictions to ActiveTension typed dicts
    tensions: list[ActiveTension] = []
    for i, c in enumerate(contradictions):
        tension_id = hashlib.sha256(
            f"{c.get('claim_a', '')}{c.get('claim_b', '')}".encode()
        ).hexdigest()[:12]

        tensions.append(
            ActiveTension(
                id=f"tension-{tension_id}",
                claim_a=ClaimVerdict(
                    claim_id=f"cl-{tension_id}-a",
                    claim_text=c.get("claim_a", ""),
                    grounding_score=0.0,
                    grounding_method="pending",
                    authority_score=0.0,
                    authority_level="unknown",
                    contradicted=True,
                ),
                claim_b=ClaimVerdict(
                    claim_id=f"cl-{tension_id}-b",
                    claim_text=c.get("claim_b", ""),
                    grounding_score=0.0,
                    grounding_method="pending",
                    authority_score=0.0,
                    authority_level="unknown",
                    contradicted=True,
                ),
                severity=c.get("severity", "nuanced"),
                authority_differential=0.0,
                resolution_hint=f"topic: {c.get('topic', 'unknown')}",
            )
        )

    return {
        "contradictions": contradictions,
        "active_tensions": tensions,
        "token_usage": [usage] if usage else [],
    }


# ============================================================
# Coverage Judge
# ============================================================


def coverage_judge(
    scored_documents: list[ScoredDocument],
    research_question: str,
    sub_queries: list[dict],
    diversity_metrics: dict | None = None,
) -> dict[str, Any]:
    """Assess facet coverage and identify gaps.

    Deterministic — zero LLM calls. Reuses scoring/diversity.py.

    Returns:
        {
            "facets": [Facet],
            "coverage_map": CoverageMap,
            "next_wave_queries": [SubQuery-like dicts],
        }
    """
    # Generate facets from sub-queries (each unique query topic = one facet)
    facets: list[Facet] = []
    seen_topics: set[str] = set()
    for sq in sub_queries:
        q = sq.get("question", sq.get("query", ""))
        if q and q.lower() not in seen_topics:
            seen_topics.add(q.lower())
            fid = hashlib.sha256(q.encode()).hexdigest()[:10]
            facets.append(
                Facet(
                    id=f"facet-{fid}",
                    question=q,
                    weight=1.0 / max(len(sub_queries), 1),
                )
            )

    # Compute per-facet coverage by checking which facets have matching docs
    facet_coverage: dict[str, float] = {}
    doc_titles = {d.get("title", "").lower() for d in scored_documents}

    for facet in facets:
        # Simple keyword overlap: how many doc titles share terms with facet question
        facet_terms = set(facet["question"].lower().split())
        if not facet_terms:
            facet_coverage[facet["id"]] = 0.0
            continue

        matches = 0
        for title in doc_titles:
            title_terms = set(title.split())
            overlap = len(facet_terms & title_terms)
            if overlap >= max(1, len(facet_terms) // 3):
                matches += 1

        # Denominator: expected matches per facet, bounded [3, 10]
        # Floor 3 prevents inflation; ceiling 10 prevents deflation with many docs
        raw_expected = len(scored_documents) // max(len(facets), 1)
        expected_matches = max(3, min(10, raw_expected))
        coverage = min(1.0, matches / expected_matches)
        facet_coverage[facet["id"]] = round(coverage, 4)

    # Compute diversity if not provided
    if diversity_metrics is None and scored_documents:
        diversity_metrics = compute_diversity(scored_documents)

    # Overall coverage = weighted average of facet coverages
    if facets:
        total_weight = sum(f["weight"] for f in facets)
        overall = sum(facet_coverage.get(f["id"], 0.0) * f["weight"] for f in facets) / max(
            total_weight, 0.001
        )
    else:
        overall = 0.0

    # Identify uncovered facets
    uncovered = [f["question"] for f in facets if facet_coverage.get(f["id"], 0) < 0.3]

    # Identify under-represented perspectives from diversity
    under_rep: list[str] = []
    if diversity_metrics:
        auth_dist = diversity_metrics.get("authority_distribution", {})
        if auth_dist.get("institutional", 0) == 0:
            under_rep.append("institutional/academic sources")
        if auth_dist.get("professional", 0) == 0:
            under_rep.append("professional/industry sources")

    coverage_map = CoverageMap(
        facet_coverage=facet_coverage,
        overall_coverage=round(overall, 4),
        uncovered_facets=uncovered,
        under_represented_perspectives=under_rep,
    )

    # Generate next-wave queries for uncovered facets
    next_queries: list[dict] = []
    for facet_q in uncovered[:3]:  # cap at 3 follow-up queries
        qid = hashlib.sha256(f"wave-{facet_q}".encode()).hexdigest()[:10]
        next_queries.append(
            {
                "id": f"sq-wave-{qid}",
                "question": facet_q,
                "search_backends": ["searxng"],
            }
        )

    return {
        "facets": facets,
        "coverage_map": coverage_map,
        "next_wave_queries": next_queries,
    }

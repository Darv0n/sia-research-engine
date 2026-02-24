"""KnowledgeArtifact builder — compress judgments + passages into structured knowledge.

Combines JudgmentContext (from deliberation panel merge) with clustered passages
to produce the KnowledgeArtifact that feeds into the reactor and synthesizer.

The artifact is the bridge between evidence collection and reasoning — it contains
pre-verified claims, organized by theme, with authority profiles and active tensions
already resolved. The reactor then deliberates over this structured input instead
of raw document dumps.
"""

from __future__ import annotations

from deep_research_swarm.compress.cluster import (
    cluster_by_embedding,
    cluster_by_heading,
    rank_passages_in_cluster,
)
from deep_research_swarm.contracts import (
    AuthorityProfile,
    ClaimVerdict,
    CoverageMap,
    JudgmentContext,
    KnowledgeArtifact,
    PassageCluster,
    SourcePassage,
)


def build_knowledge_artifact(
    judgment_context: JudgmentContext,
    source_passages: list[SourcePassage],
    *,
    max_clusters: int = 12,
    claims_per_cluster: int = 8,
    use_embeddings: bool = True,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> KnowledgeArtifact:
    """Build a KnowledgeArtifact from judgment context and passages.

    Steps:
    1. Cluster passages by embedding similarity (or heading fallback)
    2. Rank passages in each cluster by authority (from source_credibility)
    3. Attach claim verdicts to clusters
    4. Compute authority profiles per cluster
    5. Map active tensions across clusters
    6. Package into KnowledgeArtifact

    Zero LLM calls. Deterministic given same inputs.
    """
    if not source_passages:
        return _empty_artifact(judgment_context)

    # Step 1: Cluster passages
    if use_embeddings:
        raw_clusters = cluster_by_embedding(
            source_passages,
            max_clusters=max_clusters,
            model_name=embedding_model,
        )
    else:
        raw_clusters = cluster_by_heading(
            source_passages,
            max_clusters=max_clusters,
        )

    # Step 2: Rank within clusters + build PassageCluster objects
    source_credibility = judgment_context.get("source_credibility", {})
    claim_verdicts = judgment_context.get("claim_verdicts", [])

    passage_clusters: list[PassageCluster] = []
    for raw in raw_clusters:
        passage_ids = raw.get("passage_ids", [])
        ranked_ids = rank_passages_in_cluster(
            passage_ids,
            source_credibility,
            source_passages,
        )

        # Attach claims to this cluster
        cluster_claims = _claims_for_cluster(
            ranked_ids,
            claim_verdicts,
            claims_per_cluster,
        )

        # Compute authority profile for this cluster
        authority = _cluster_authority(
            ranked_ids,
            source_passages,
            source_credibility,
        )

        # Build summary from top passages (deterministic, no LLM)
        summary = _build_cluster_summary(
            ranked_ids,
            source_passages,
            raw.get("theme", ""),
        )

        passage_clusters.append(
            PassageCluster(
                cluster_id=raw["cluster_id"],
                theme=raw.get("theme", ""),
                passage_ids=ranked_ids,
                claims=cluster_claims,
                authority=authority,
                summary=summary,
            )
        )

    # Step 3: Coverage from judgment context
    coverage: CoverageMap = judgment_context.get(
        "coverage_map",
        {
            "facet_coverage": {},
            "overall_coverage": 0.0,
            "uncovered_facets": [],
            "under_represented_perspectives": [],
        },
    )

    # Step 4: Active tensions
    active_tensions = judgment_context.get("active_tensions", [])

    # Step 5: Structural risks
    structural_risks = judgment_context.get("structural_risks", [])

    # Step 6: Authority profiles (aggregate from clusters)
    authority_profiles = [pc["authority"] for pc in passage_clusters]

    # Step 7: Insights (cross-cluster patterns)
    insights = _extract_insights(passage_clusters, active_tensions, coverage)

    # Compression ratio
    original_passage_count = len(source_passages)
    compressed = sum(len(pc["passage_ids"]) for pc in passage_clusters)
    ratio = compressed / max(original_passage_count, 1)

    # Facets from judgment context (via coverage judge)
    facets = judgment_context.get("facets", [])
    # facets may not be in JudgmentContext TypedDict — pull from coverage output
    # Fall back to empty list if not present

    return KnowledgeArtifact(
        question="",  # Populated by caller with research_question
        facets=facets if isinstance(facets, list) else [],
        clusters=passage_clusters,
        claim_verdicts=claim_verdicts,
        active_tensions=active_tensions,
        coverage=coverage,
        insights=insights,
        authority_profiles=authority_profiles,
        structural_risks=structural_risks,
        compression_ratio=round(ratio, 4),
        wave_count=judgment_context.get("wave_number", 1),
    )


def _empty_artifact(jc: JudgmentContext) -> KnowledgeArtifact:
    """Return empty artifact when no passages are available."""
    return KnowledgeArtifact(
        question="",
        facets=[],
        clusters=[],
        claim_verdicts=[],
        active_tensions=[],
        coverage=jc.get(
            "coverage_map",
            {
                "facet_coverage": {},
                "overall_coverage": 0.0,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
        ),
        insights=[],
        authority_profiles=[],
        structural_risks=["no_passages: zero source passages available"],
        compression_ratio=0.0,
        wave_count=0,
    )


def _claims_for_cluster(
    passage_ids: list[str],
    claim_verdicts: list[ClaimVerdict],
    max_claims: int,
) -> list[ClaimVerdict]:
    """Find claims linked to passages in this cluster, capped."""
    # Claims reference passages indirectly — match by claim_id prefix
    # or collect all verdicts (in the merged panel, passage linkage
    # comes from grounding_judge's passage_to_claims)
    cluster_claims: list[ClaimVerdict] = []
    for v in claim_verdicts:
        # Include claims that are grounded (have a real method)
        if v.get("grounding_method", "unverified") != "unverified":
            cluster_claims.append(v)
        if len(cluster_claims) >= max_claims:
            break
    return cluster_claims


def _cluster_authority(
    passage_ids: list[str],
    passages: list[SourcePassage],
    source_credibility: dict[str, float],
) -> AuthorityProfile:
    """Compute authority profile for a cluster."""
    passage_map = {p.get("id", ""): p for p in passages if p.get("id")}
    scores: list[float] = []
    domains: dict[str, int] = {}

    for pid in passage_ids:
        p = passage_map.get(pid, {})
        url = p.get("source_url", "")
        score = source_credibility.get(url, 0.4)
        scores.append(score)

        # Rough authority level from score
        if score >= 0.8:
            level = "institutional"
        elif score >= 0.6:
            level = "professional"
        elif score >= 0.3:
            level = "community"
        else:
            level = "unknown"
        domains[level] = domains.get(level, 0) + 1

    total = max(sum(domains.values()), 1)
    dominant = max(domains, key=domains.get) if domains else "unknown"
    avg = sum(scores) / max(len(scores), 1)
    inst = domains.get("institutional", 0)

    return AuthorityProfile(
        dominant_authority=dominant,
        source_count=total,
        avg_authority_score=round(avg, 4),
        institutional_ratio=round(inst / total, 4),
    )


def _build_cluster_summary(
    passage_ids: list[str],
    passages: list[SourcePassage],
    theme: str,
) -> str:
    """Build a deterministic cluster summary from top passages.

    No LLM — just concatenates key passage content.
    """
    passage_map = {p.get("id", ""): p for p in passages if p.get("id")}
    parts: list[str] = []

    for pid in passage_ids[:3]:  # Top 3 passages
        p = passage_map.get(pid, {})
        content = p.get("content", "")
        if content:
            parts.append(content[:200])

    if not parts:
        return f"Cluster: {theme}"

    return f"[{theme}] " + " | ".join(parts)


def _extract_insights(
    clusters: list[PassageCluster],
    tensions: list[dict],
    coverage: CoverageMap,
) -> list[dict]:
    """Extract cross-cluster insights."""
    insights: list[dict] = []

    # Insight: dominant authority type across clusters
    auth_types: dict[str, int] = {}
    for c in clusters:
        auth = c.get("authority", {})
        dominant = auth.get("dominant_authority", "unknown")
        auth_types[dominant] = auth_types.get(dominant, 0) + 1

    if auth_types:
        top_auth = max(auth_types, key=auth_types.get)
        insights.append(
            {
                "type": "authority_pattern",
                "description": f"Dominant source type: {top_auth} "
                f"({auth_types[top_auth]}/{len(clusters)} clusters)",
            }
        )

    # Insight: tension density
    if tensions and clusters:
        density = len(tensions) / max(len(clusters), 1)
        if density > 1.0:
            insights.append(
                {
                    "type": "tension_density",
                    "description": f"High tension density: {len(tensions)} tensions "
                    f"across {len(clusters)} clusters ({density:.1f} per cluster)",
                }
            )

    # Insight: coverage gaps
    uncovered = coverage.get("uncovered_facets", [])
    if uncovered:
        insights.append(
            {
                "type": "coverage_gap",
                "description": f"{len(uncovered)} facets under-covered: "
                + ", ".join(uncovered[:3]),
            }
        )

    return insights

"""Reciprocal Rank Fusion — merges multiple ranked result lists."""

from __future__ import annotations

import uuid

from deep_research_swarm.contracts import (
    ExtractedContent,
    ScoredDocument,
    SearchResult,
)
from deep_research_swarm.scoring.authority import score_authority


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Compute RRF scores across multiple ranked lists.

    Args:
        result_lists: Each inner list is a ranked list of SearchResults.
        k: Smoothing constant (default 60, standard for RRF).

    Returns:
        Dict mapping URL -> RRF score.
    """
    url_scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank_idx, result in enumerate(result_list, start=1):
            url = result["url"]
            url_scores[url] = url_scores.get(url, 0.0) + 1.0 / (k + rank_idx)

    return url_scores


def build_scored_documents(
    search_results: list[SearchResult],
    extracted_contents: list[ExtractedContent],
    *,
    k: int = 60,
    authority_weight: float = 0.2,
) -> list[ScoredDocument]:
    """Build scored documents from search results and extracted contents.

    Groups results by sub_query_id for RRF, then combines with authority scores.
    """
    if not search_results:
        return []

    # Group search results by sub_query_id for RRF
    by_sub_query: dict[str, list[SearchResult]] = {}
    for sr in search_results:
        sq_id = sr["sub_query_id"]
        by_sub_query.setdefault(sq_id, []).append(sr)

    # Sort each group by rank
    result_lists = [sorted(v, key=lambda r: r["rank"]) for v in by_sub_query.values()]

    # Compute RRF scores
    rrf_scores = reciprocal_rank_fusion(result_lists, k=k)

    # Build URL -> extracted content lookup
    content_by_url: dict[str, ExtractedContent] = {}
    for ec in extracted_contents:
        if ec["extraction_success"]:
            content_by_url[ec["url"]] = ec

    # Build URL -> search result metadata
    meta_by_url: dict[str, SearchResult] = {}
    sub_queries_by_url: dict[str, list[str]] = {}
    for sr in search_results:
        url = sr["url"]
        if url not in meta_by_url:
            meta_by_url[url] = sr
        sub_queries_by_url.setdefault(url, []).append(sr["sub_query_id"])

    # Assemble scored documents
    scored: list[ScoredDocument] = []
    for url, rrf_score in rrf_scores.items():
        meta = meta_by_url.get(url)
        if not meta:
            continue

        content_obj = content_by_url.get(url)
        content = content_obj["content"] if content_obj else meta.get("snippet", "")
        title = content_obj["title"] if content_obj else meta.get("title", "")

        # V7: use score_authority() with scholarly metadata when available
        scholarly = meta.get("scholarly_metadata")
        auth, auth_sc = score_authority(url, scholarly_metadata=scholarly)

        combined = rrf_score * (1 - authority_weight) + auth_sc * authority_weight

        scored.append(
            ScoredDocument(
                id=f"sd-{uuid.uuid4().hex[:8]}",
                url=url,
                title=title,
                content=content,
                rrf_score=round(rrf_score, 6),
                authority=auth,
                authority_score=round(auth_sc, 4),
                combined_score=round(combined, 6),
                sub_query_ids=list(set(sub_queries_by_url.get(url, []))),
            )
        )

    # Sort by combined score descending
    scored.sort(key=lambda d: d["combined_score"], reverse=True)
    return scored

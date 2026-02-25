"""Passage clustering — group source passages by theme.

Two strategies:
  1. Embedding-based (fastembed) — semantic similarity clustering
  2. Heading-based (fallback) — group by section heading / source domain

Both are deterministic given the same input. Zero LLM calls.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlparse

from deep_research_swarm.contracts import SourcePassage


def cluster_by_embedding(
    passages: list[SourcePassage],
    *,
    max_clusters: int = 12,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[dict[str, Any]]:
    """Cluster passages by embedding similarity.

    Uses fastembed (optional dep). Falls back to heading-based clustering
    if fastembed is not installed.

    Returns list of cluster dicts:
        [{"cluster_id": str, "theme": str, "passage_ids": [str]}]
    """
    try:
        from deep_research_swarm.scoring.embedding_grounding import (
            get_embedding_provider,
        )

        provider = get_embedding_provider(model_name)
        if provider is None:
            return cluster_by_heading(passages, max_clusters=max_clusters)
    except ImportError:
        return cluster_by_heading(passages, max_clusters=max_clusters)

    if not passages:
        return []

    # Encode all passages
    texts = [_passage_text(p) for p in passages]
    embeddings = [provider.encode(t) for t in texts]

    # Simple greedy clustering: assign each passage to nearest centroid
    # Start with first passage as centroid, add new centroids when
    # similarity to all existing centroids is below threshold
    similarity_threshold = 0.75
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []

    for i, emb in enumerate(embeddings):
        best_cluster = -1
        best_sim = -1.0

        for j, centroid in enumerate(centroids):
            sim = provider.similarity(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = j

        if best_sim >= similarity_threshold and best_cluster >= 0:
            clusters[best_cluster].append(i)
        elif len(clusters) < max_clusters:
            clusters.append([i])
            centroids.append(emb)
        else:
            # Assign to closest cluster if at max
            if best_cluster >= 0:
                clusters[best_cluster].append(i)

    return _format_clusters(passages, clusters)


def cluster_by_heading(
    passages: list[SourcePassage],
    *,
    max_clusters: int = 12,
) -> list[dict[str, Any]]:
    """Cluster passages by heading / source domain (fallback).

    Groups passages sharing the same heading prefix or domain.
    Deterministic, zero dependencies.
    """
    if not passages:
        return []

    # Group by heading first, then by domain
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(passages):
        heading = p.get("heading_context", "")
        if heading:
            key = _normalize_heading(heading)
        else:
            url = p.get("source_url", "")
            key = _extract_domain(url) or "ungrouped"

        groups.setdefault(key, []).append(i)

    # Merge small groups
    clusters: list[list[int]] = []
    overflow: list[int] = []

    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    for group in sorted_groups:
        if len(group) >= 2 and len(clusters) < max_clusters:
            clusters.append(group)
        else:
            overflow.extend(group)

    # Distribute overflow into existing clusters or create final cluster
    if overflow:
        if len(clusters) >= max_clusters:
            clusters[-1].extend(overflow)
        else:
            clusters.append(overflow)

    return _format_clusters(passages, clusters)


def rank_passages_in_cluster(
    cluster_passage_ids: list[str],
    source_credibility: dict[str, float],
    passages: list[SourcePassage],
) -> list[str]:
    """Rank passage IDs within a cluster by merged credibility score.

    Higher authority score = higher rank. Returns sorted passage IDs.
    """
    passage_map = {p.get("id", ""): p for p in passages if p.get("id")}

    def score(pid: str) -> float:
        p = passage_map.get(pid, {})
        url = p.get("source_url", "")
        return source_credibility.get(url, 0.4)

    return sorted(cluster_passage_ids, key=score, reverse=True)


# ============================================================
# Internal helpers
# ============================================================


def _passage_text(p: SourcePassage) -> str:
    """Extract representative text from a passage for embedding."""
    heading = p.get("heading_context", "")
    content = p.get("content", "")
    # Combine heading + first 500 chars of content
    text = f"{heading}: {content[:500]}" if heading else content[:500]
    return text or "empty"


def _normalize_heading(heading: str) -> str:
    """Normalize heading for grouping."""
    h = heading.lower().strip()
    h = re.sub(r"\d+\.\s*", "", h)  # strip numbering
    h = re.sub(r"\s+", " ", h)
    # Take first 3 words for grouping
    words = h.split()[:3]
    return " ".join(words) if words else "ungrouped"


def _extract_domain(url: str) -> str:
    """Extract registered domain from URL."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower()
    except Exception:
        return ""


def _format_clusters(
    passages: list[SourcePassage],
    clusters: list[list[int]],
) -> list[dict[str, Any]]:
    """Format cluster indices into typed cluster dicts."""
    result: list[dict[str, Any]] = []
    for i, indices in enumerate(clusters):
        passage_ids = [passages[idx].get("id", f"p-{idx}") for idx in indices]
        # Theme from first passage heading or domain
        first = passages[indices[0]] if indices else {}
        theme = first.get("heading_context", _extract_domain(first.get("source_url", "")))
        cid = hashlib.sha256(",".join(sorted(passage_ids)).encode()).hexdigest()[:10]

        result.append(
            {
                "cluster_id": f"cluster-{cid}",
                "theme": theme or f"cluster-{i}",
                "passage_ids": passage_ids,
            }
        )
    return result

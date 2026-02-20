"""Text similarity utilities — Jaccard scoring and duplicate detection."""

from __future__ import annotations


def jaccard_score(a: str, b: str) -> float:
    """Normalized token overlap (Jaccard similarity) between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def is_duplicate(new_query: str, existing: list[str], threshold: float = 0.7) -> bool:
    """Check if a query is a near-duplicate of any existing query.

    Uses normalized token overlap (Jaccard similarity) as a fast heuristic,
    plus substring containment as a fallback.
    """
    new_tokens = set(new_query.lower().split())
    if not new_tokens:
        return True

    for existing_q in existing:
        existing_tokens = set(existing_q.lower().split())
        if not existing_tokens:
            continue

        intersection = new_tokens & existing_tokens
        union = new_tokens | existing_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        if jaccard >= threshold:
            return True

        # Also catch substring containment
        if new_query.lower() in existing_q.lower():
            return True
        if existing_q.lower() in new_query.lower():
            return True

    return False

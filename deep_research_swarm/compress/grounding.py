"""Claim verification cascade — 3-tier grounding check.

Tier 1: Embedding similarity (fastembed, if available)
Tier 2: Jaccard token overlap (pure Python, always available)

Each tier returns (grounded: bool, score: float, method: str).
The cascade tries higher tiers first and falls back on failure.

Deterministic. Zero LLM calls.
"""

from __future__ import annotations

from deep_research_swarm.contracts import SourcePassage


def verify_claim(
    claim_text: str,
    passage: SourcePassage,
    *,
    embedding_provider: object | None = None,
    embedding_threshold: float = 0.7,
    jaccard_threshold: float = 0.3,
) -> tuple[bool, float, str]:
    """Verify a claim against a source passage using cascade.

    Tries embedding similarity first, falls back to Jaccard.

    Returns (grounded, score, method).
    """
    # Tier 1: Embedding similarity
    if embedding_provider is not None:
        try:
            from deep_research_swarm.scoring.embedding_grounding import (
                verify_grounding_embedding,
            )

            grounded, score, method = verify_grounding_embedding(
                claim_text,
                passage,
                embedding_provider,
                threshold=embedding_threshold,
            )
            return grounded, score, method
        except Exception:
            pass  # Fall through to Jaccard

    # Tier 2: Jaccard token overlap (always available)
    from deep_research_swarm.scoring.grounding import verify_grounding

    return verify_grounding(
        claim_text,
        passage,
        threshold=jaccard_threshold,
    )


def verify_claims_batch(
    claims: list[dict],
    passages: list[SourcePassage],
    claim_to_passages: dict[str, list[str]],
    *,
    embedding_provider: object | None = None,
) -> list[dict]:
    """Verify a batch of claims against their linked passages.

    Returns list of verification results:
        [{"claim_id": str, "grounded": bool, "score": float, "method": str}]
    """
    passage_map = {p.get("id", ""): p for p in passages if p.get("id")}
    results: list[dict] = []

    for claim in claims:
        claim_id = claim.get("claim_id", claim.get("id", ""))
        claim_text = claim.get("claim_text", claim.get("text", ""))
        linked = claim_to_passages.get(claim_id, [])

        best_score = 0.0
        best_method = "unverified"
        grounded = False

        for pid in linked:
            passage = passage_map.get(pid)
            if not passage:
                continue

            ok, score, method = verify_claim(
                claim_text,
                passage,
                embedding_provider=embedding_provider,
            )
            if score > best_score:
                best_score = score
                best_method = method
                grounded = ok

        results.append(
            {
                "claim_id": claim_id,
                "grounded": grounded,
                "score": round(best_score, 4),
                "method": best_method,
            }
        )

    return results

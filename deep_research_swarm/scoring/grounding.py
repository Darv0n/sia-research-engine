"""Mechanical grounding verification — LLM-free, deterministic (I2).

All functions in this module are sub-millisecond per call.
verify_grounding() returns method="jaccard_v1" (OE5, D7) so V8
can add embedding-based verification with both methods coexisting.

threshold=0.3 is intentionally permissive (D2): catches citation
fraud, not paraphrasing. Goodhart's Law risk is real — do not
raise threshold without calibration (I3).
"""

from __future__ import annotations

import re

from deep_research_swarm.contracts import SectionOutline, SourcePassage
from deep_research_swarm.scoring.embedding_grounding import (
    EmbeddingProvider,
    verify_grounding_embedding,
)

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "and",
        "or",
        "but",
        "for",
        "with",
        "at",
        "by",
        "from",
        "that",
        "this",
        "it",
        "be",
        "as",
        "on",
        "not",
        "have",
        "had",
        "has",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_relevant_passages(
    query: str,
    passages: list[SourcePassage],
    *,
    top_k: int = 10,
) -> list[tuple[SourcePassage, float]]:
    """Find passages most relevant to a query using Jaccard + exact phrase boost.

    Returns top-k (passage, score) pairs sorted by score descending.
    """
    if not passages or not query.strip():
        return []

    query_tokens = _tokenize(query)
    query_lower = query.lower()
    scored: list[tuple[SourcePassage, float]] = []

    for passage in passages:
        passage_tokens = _tokenize(passage["content"])
        score = _jaccard(query_tokens, passage_tokens)

        # Exact phrase boost: if query substring appears in passage
        if len(query_lower) > 10 and query_lower[:30] in passage["content"].lower():
            score = min(1.0, score + 0.15)

        scored.append((passage, round(score, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def assign_passages_to_sections(
    outline: list[SectionOutline],
    passages: list[SourcePassage],
    *,
    max_passages_per_section: int = 8,
) -> dict[str, list[SourcePassage]]:
    """Assign passages to sections based on source_ids and key_claims relevance.

    For each section: filter to assigned source_ids, score against
    key_claims, return top-N passages.
    """
    if not outline or not passages:
        return {}

    # Index passages by source_id
    passages_by_source: dict[str, list[SourcePassage]] = {}
    for p in passages:
        sid = p["source_id"]
        if sid not in passages_by_source:
            passages_by_source[sid] = []
        passages_by_source[sid].append(p)

    result: dict[str, list[SourcePassage]] = {}

    for section in outline:
        heading = section["heading"]
        source_ids = section.get("source_ids", [])
        key_claims = section.get("key_claims", [])

        # Gather candidate passages from assigned sources
        candidates: list[SourcePassage] = []
        for sid in source_ids:
            candidates.extend(passages_by_source.get(sid, []))

        # If no source_ids match, fall back to all passages
        if not candidates:
            candidates = list(passages)

        if not key_claims:
            # No claims to score against — take first N by position
            candidates.sort(key=lambda p: p["position"])
            result[heading] = candidates[:max_passages_per_section]
            continue

        # Score each candidate against key_claims
        claims_tokens = _tokenize(" ".join(key_claims))
        scored: list[tuple[SourcePassage, float]] = []
        for p in candidates:
            p_tokens = _tokenize(p["content"])
            score = _jaccard(claims_tokens, p_tokens)
            scored.append((p, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result[heading] = [p for p, _ in scored[:max_passages_per_section]]

    return result


def verify_grounding(
    claim: str,
    cited_passage: SourcePassage,
    *,
    threshold: float = 0.3,
    method: str = "jaccard_v1",
) -> tuple[bool, float, str]:
    """Mechanically verify a claim is grounded in a cited passage (I2, D2, D7).

    Returns (is_grounded, similarity_score, method).
    NOT an LLM call. Deterministic keyword/token overlap.
    threshold=0.3 is intentionally low: catches citation fraud,
    not paraphrasing. Goodhart's Law risk is real — do not raise
    threshold without calibration (I3).
    """
    claim_tokens = _tokenize(claim)
    passage_tokens = _tokenize(cited_passage["content"])
    score = _jaccard(claim_tokens, passage_tokens)
    return (score >= threshold, round(score, 4), method)


def second_pass_grounding(
    claim: str,
    borderline_passage: SourcePassage,
    accepted_passages: list[SourcePassage],
    *,
    borderline_floor: float = 0.15,
    neighborhood_threshold: float = 0.25,
    method: str = "neighborhood_v1",
) -> tuple[bool, float, str]:
    """Semantic neighborhood reassessment for borderline passages (V8, I7).

    When a source is borderline (Jaccard score between borderline_floor and
    the grounding threshold), check if ACCEPTED passages are topically related.
    If 2+ accepted passages share vocabulary with the borderline passage,
    it's in the same semantic neighborhood and gets rescued.

    Returns (is_rescued, adjusted_score, method).
    Purely additive — never downgrades a passage that first-pass accepted.
    """
    claim_tokens = _tokenize(claim)
    borderline_tokens = _tokenize(borderline_passage["content"])
    base_score = _jaccard(claim_tokens, borderline_tokens)

    # Below the floor — too far gone, don't attempt rescue
    if base_score < borderline_floor:
        return (False, round(base_score, 4), method)

    # Count accepted passages that share vocabulary with borderline passage
    neighbors = 0
    for accepted in accepted_passages:
        accepted_tokens = _tokenize(accepted["content"])
        overlap = _jaccard(borderline_tokens, accepted_tokens)
        if overlap >= neighborhood_threshold:
            neighbors += 1

    is_rescued = neighbors >= 2
    if is_rescued:
        # Boost score proportional to neighborhood density, cap at 0.35
        adjusted_score = min(0.35, base_score + 0.05 * neighbors)
    else:
        adjusted_score = base_score

    return (is_rescued, round(adjusted_score, 4), method)


def compute_section_grounding_score(
    section_content: str,
    section_citations: list[str],
    passages: list[SourcePassage],
    citation_to_passage_map: dict[str, list[str]],
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[float, list[dict]]:
    """Compute grounding score for a section (OE3, D4).

    Returns (grounding_score, claim_details).
    grounding_score = count(grounded) / count(cited_claims)
    claim_details: [{claim, citation_id, passage_id, grounded, similarity, method}]
    """
    if not section_citations or not passages:
        return (0.0, [])

    # Build passage lookup
    passage_by_id: dict[str, SourcePassage] = {p["id"]: p for p in passages}

    # Extract claim-like sentences from section content
    # A claim is a sentence containing a citation marker [N]
    citation_re = re.compile(r"\[(\d+)\]")
    sentences = re.split(r"(?<=[.!?])\s+", section_content)

    claim_details: list[dict] = []
    grounded_count = 0
    total_claims = 0

    for sentence in sentences:
        refs = citation_re.findall(sentence)
        if not refs:
            continue

        # Clean the sentence (remove citation markers for comparison)
        clean_claim = citation_re.sub("", sentence).strip()
        if not clean_claim:
            continue

        for ref in refs:
            citation_id = f"[{ref}]"
            if citation_id not in section_citations:
                continue

            total_claims += 1
            passage_ids = citation_to_passage_map.get(citation_id, [])

            best_grounded = False
            best_similarity = 0.0
            best_passage_id = ""
            best_method = "jaccard_v1"

            for pid in passage_ids:
                passage = passage_by_id.get(pid)
                if passage is None:
                    continue
                grounded, similarity, method = verify_grounding(clean_claim, passage)
                if similarity > best_similarity:
                    best_grounded = grounded
                    best_similarity = similarity
                    best_passage_id = pid
                    best_method = method

                # Embedding fallback: if Jaccard didn't ground and provider available
                if not grounded and embedding_provider is not None:
                    emb_grounded, emb_sim, emb_method = verify_grounding_embedding(
                        clean_claim, passage, embedding_provider
                    )
                    if emb_sim > best_similarity:
                        best_grounded = emb_grounded
                        best_similarity = emb_sim
                        best_passage_id = pid
                        best_method = emb_method

            if best_grounded:
                grounded_count += 1

            claim_details.append(
                {
                    "claim": clean_claim,
                    "citation_id": citation_id,
                    "passage_id": best_passage_id,
                    "grounded": best_grounded,
                    "similarity": best_similarity,
                    "method": best_method,
                }
            )

    if total_claims == 0:
        return (0.0, claim_details)

    # --- Second pass: rescue borderline claims via semantic neighborhood ---
    # Collect passages that were accepted (grounded) in first pass
    accepted_pids = {d["passage_id"] for d in claim_details if d["grounded"] and d["passage_id"]}
    accepted_passages_list = [passage_by_id[pid] for pid in accepted_pids if pid in passage_by_id]

    if accepted_passages_list:
        for detail in claim_details:
            if detail["grounded"] or not detail["passage_id"]:
                continue  # already grounded or no passage to test
            # Only attempt rescue for borderline scores (>= 0.15)
            if detail["similarity"] < 0.15:
                continue

            borderline_p = passage_by_id.get(detail["passage_id"])
            if borderline_p is None:
                continue

            rescued, adj_score, adj_method = second_pass_grounding(
                detail["claim"],
                borderline_p,
                accepted_passages_list,
            )
            if rescued:
                detail["grounded"] = True
                detail["similarity"] = adj_score
                detail["method"] = adj_method
                grounded_count += 1

    grounding_score = round(grounded_count / total_claims, 4)
    return (grounding_score, claim_details)

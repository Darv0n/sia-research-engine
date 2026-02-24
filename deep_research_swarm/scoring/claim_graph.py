"""Claim graph — extract claims from sections, link to source passages (V8, PR-08).

Populates SourcePassage.claim_ids (OE1 stub) and builds a structured
claim-to-passage mapping for provenance tracking and per-claim confidence.

All functions are LLM-free and deterministic.
"""

from __future__ import annotations

import hashlib
import re

from deep_research_swarm.contracts import SectionDraft, SourcePassage


def _claim_id(section_id: str, idx: int) -> str:
    """Generate a deterministic claim ID from section ID and claim index."""
    raw = f"{section_id}:claim:{idx}"
    return f"cl-{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def extract_claims_from_section(
    section: SectionDraft,
) -> list[dict]:
    """Extract individual claims from a section's content.

    A claim is a sentence containing one or more citation markers [N].

    Returns list of:
        {"id": "cl-...", "text": "clean claim text", "citation_ids": ["[1]", "[2]"],
         "section_id": "sec-...", "section_heading": "..."}
    """
    content = section["content"]
    section_id = section["id"]
    heading = section["heading"]

    citation_re = re.compile(r"\[(\d+)\]")
    sentences = re.split(r"(?<=[.!?])\s+", content)

    claims: list[dict] = []
    idx = 0

    for sentence in sentences:
        refs = citation_re.findall(sentence)
        if not refs:
            continue

        clean_text = citation_re.sub("", sentence).strip()
        if not clean_text:
            continue

        citation_ids = [f"[{r}]" for r in refs]
        claims.append(
            {
                "id": _claim_id(section_id, idx),
                "text": clean_text,
                "citation_ids": citation_ids,
                "section_id": section_id,
                "section_heading": heading,
            }
        )
        idx += 1

    return claims


def link_claims_to_passages(
    claims: list[dict],
    citation_to_passage_map: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Map claim IDs to passage IDs they cite.

    Returns {claim_id: [passage_id, ...]}.
    """
    claim_to_passages: dict[str, list[str]] = {}

    for claim in claims:
        passage_ids: list[str] = []
        for cid in claim["citation_ids"]:
            passage_ids.extend(citation_to_passage_map.get(cid, []))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for pid in passage_ids:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        claim_to_passages[claim["id"]] = unique

    return claim_to_passages


def populate_claim_ids(
    passages: list[SourcePassage],
    claims: list[dict],
    citation_to_passage_map: dict[str, list[str]],
) -> list[SourcePassage]:
    """Populate claim_ids on source passages (OE1).

    For each passage, find all claims that cite it (via citation_to_passage_map)
    and add their claim IDs to the passage's claim_ids list.

    Returns a new list of passages with claim_ids populated.
    Does NOT mutate the input passages.
    """
    # Build reverse map: passage_id -> [claim_id]
    claim_to_passages = link_claims_to_passages(claims, citation_to_passage_map)
    passage_to_claims: dict[str, list[str]] = {}
    for claim_id, pids in claim_to_passages.items():
        for pid in pids:
            if pid not in passage_to_claims:
                passage_to_claims[pid] = []
            passage_to_claims[pid].append(claim_id)

    # Build updated passages (shallow copy + overwrite claim_ids)
    updated: list[SourcePassage] = []
    for p in passages:
        claim_ids = passage_to_claims.get(p["id"], [])
        new_p = dict(p)
        new_p["claim_ids"] = claim_ids
        updated.append(new_p)  # type: ignore[arg-type]

    return updated

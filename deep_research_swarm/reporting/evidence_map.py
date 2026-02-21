"""Evidence map — maps report claims to their supporting sources."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deep_research_swarm.contracts import Citation, SectionDraft


# Matches sentences containing at least one [N] citation marker
_CITATION_RE = re.compile(r"\[(\d+)\]")
# Crude sentence splitter: split on ". " or ".\n" but keep [N] refs intact
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if len(s.strip()) > 20]


def _escape_pipe(text: str) -> str:
    """Escape pipe characters for Markdown table cells."""
    return text.replace("|", "\\|")


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def extract_claims(
    section_drafts: list[SectionDraft],
    citations: list[Citation],
    *,
    max_claims: int = 20,
) -> list[dict]:
    """Extract cited sentences from section drafts and map to citation metadata.

    Returns a list of claim dicts:
        {claim, section, citation_id, source_title, source_url, authority, confidence}
    """
    # Build citation lookup: "[1]" -> Citation
    citation_map: dict[str, Citation] = {}
    for cit in citations:
        citation_map[cit["id"]] = cit

    claims: list[dict] = []

    for section in section_drafts:
        sentences = _split_sentences(section["content"])
        for sentence in sentences:
            refs = _CITATION_RE.findall(sentence)
            if not refs:
                continue

            for ref_num in refs:
                cit_id = f"[{ref_num}]"
                cit = citation_map.get(cit_id)
                if cit is None:
                    continue

                claims.append(
                    {
                        "claim": sentence,
                        "section": section["heading"],
                        "citation_id": cit_id,
                        "source_title": cit["title"],
                        "source_url": cit["url"],
                        "authority": cit["authority"],
                        "confidence": section["confidence_score"],
                    }
                )

                if len(claims) >= max_claims:
                    return claims

    return claims


def render_evidence_map(claims: list[dict]) -> str:
    """Render claims as a Markdown table.

    Columns: Claim | Source | Authority | Confidence
    """
    if not claims:
        return ""

    lines = [
        "| Claim | Source | Authority | Confidence |",
        "|-------|--------|-----------|------------|",
    ]

    for c in claims:
        claim = _escape_pipe(_truncate(c["claim"]))
        source = _escape_pipe(c["source_title"])
        authority = c["authority"]
        if hasattr(authority, "value"):
            authority = authority.value
        confidence = f"{c['confidence']:.2f}"
        lines.append(f"| {claim} | {source} | {authority} | {confidence} |")

    return "\n".join(lines)

"""Citation management, deduplication, renumbering, and bibliography."""

from __future__ import annotations

import re

from deep_research_swarm.contracts import Citation, SectionDraft


def deduplicate_and_renumber(
    sections: list[SectionDraft],
    citations: list[Citation],
) -> tuple[list[SectionDraft], list[Citation]]:
    """Deduplicate citations by URL and renumber sequentially.

    Returns updated (sections, citations) with consistent [N] references.
    """
    if not citations:
        return sections, citations

    # Step 1: Deduplicate by URL, assign new sequential numbers
    seen_urls: dict[str, int] = {}
    unique: list[Citation] = []
    # Map old citation id -> new sequential number
    old_to_new: dict[str, str] = {}

    for cit in citations:
        url = cit["url"]
        if url not in seen_urls:
            new_num = len(unique) + 1
            seen_urls[url] = new_num
            unique.append(
                Citation(
                    id=f"[{new_num}]",
                    url=cit["url"],
                    title=cit["title"],
                    authority=cit["authority"],
                    accessed=cit["accessed"],
                    used_in_sections=list(cit["used_in_sections"]),
                )
            )
        # Map this citation's old id to the canonical new number
        new_num = seen_urls[url]
        old_to_new[cit["id"]] = f"[{new_num}]"

    # Step 2: Renumber citation refs in section content
    updated_sections: list[SectionDraft] = []
    for sec in sections:
        new_content = _renumber_content(sec["content"], old_to_new)
        new_cit_ids = list({old_to_new.get(c, c) for c in sec["citation_ids"]})

        updated_sections.append(
            SectionDraft(
                id=sec["id"],
                heading=sec["heading"],
                content=new_content,
                citation_ids=sorted(new_cit_ids),
                confidence_score=sec["confidence_score"],
                confidence_level=sec["confidence_level"],
                grader_scores=sec["grader_scores"],
            )
        )

    # Step 3: Merge used_in_sections for deduplicated citations
    for cit in unique:
        cit["used_in_sections"] = []
    for sec in updated_sections:
        for cit_id in sec["citation_ids"]:
            for cit in unique:
                if cit["id"] == cit_id and sec["heading"] not in cit["used_in_sections"]:
                    cit["used_in_sections"].append(sec["heading"])

    return updated_sections, unique


def _renumber_content(content: str, old_to_new: dict[str, str]) -> str:
    """Replace citation markers in content using old->new map.

    Handles [N] format. Uses regex to avoid partial matches
    (e.g., [10] shouldn't partially match [1]).
    """
    # Sort by length descending to replace [10] before [1]
    sorted_pairs = sorted(old_to_new.items(), key=lambda x: len(x[0]), reverse=True)

    # Use temp placeholders to avoid double-replacement
    placeholders: dict[str, str] = {}
    for i, (old, new) in enumerate(sorted_pairs):
        placeholder = f"\x00CITE{i}\x00"
        placeholders[placeholder] = new
        # Escape brackets for regex
        pattern = re.escape(old)
        content = re.sub(pattern, placeholder, content)

    # Replace placeholders with final values
    for placeholder, new in placeholders.items():
        content = content.replace(placeholder, new)

    return content


def build_bibliography(citations: list[Citation]) -> str:
    """Render citations as a numbered bibliography in Markdown.

    Expects already-deduplicated and renumbered citations.
    """
    if not citations:
        return ""

    lines: list[str] = []
    for i, cit in enumerate(citations, start=1):
        authority = cit.get("authority", "unknown")
        title = cit["title"] or cit["url"]
        lines.append(f"{i}. [{title}]({cit['url']}) — *{authority}*")

    return "\n".join(lines)

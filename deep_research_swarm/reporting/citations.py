"""Citation management and bibliography generation."""

from __future__ import annotations

from deep_research_swarm.contracts import Citation


def build_bibliography(citations: list[Citation]) -> str:
    """Render citations as a numbered bibliography in Markdown."""
    if not citations:
        return ""

    # Deduplicate by URL, keep first occurrence
    seen_urls: set[str] = set()
    unique: list[Citation] = []
    for cit in citations:
        if cit["url"] not in seen_urls:
            seen_urls.add(cit["url"])
            unique.append(cit)

    lines: list[str] = []
    for i, cit in enumerate(unique, start=1):
        authority = cit.get("authority", "unknown")
        title = cit["title"] or cit["url"]
        lines.append(f"{i}. [{title}]({cit['url']}) — *{authority}*")

    return "\n".join(lines)


def renumber_citations(
    content: str,
    old_to_new: dict[str, str],
) -> str:
    """Replace citation markers in content (e.g., [1] -> [3])."""
    result = content
    for old, new in old_to_new.items():
        result = result.replace(old, new)
    return result

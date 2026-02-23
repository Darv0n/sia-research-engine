"""Provenance section rendering for reports (V7, PR-06).

Generates a Markdown table showing source provenance chain:
URL, extractor used, source kind, and content hash.
"""

from __future__ import annotations

from deep_research_swarm.contracts import SearchResult


def render_provenance_section(search_results: list[SearchResult]) -> str:
    """Render a Markdown provenance table from search results.

    Only includes results that have provenance records attached.
    Returns empty string if no provenance data available.
    """
    rows: list[dict] = []
    seen_urls: set[str] = set()

    for sr in search_results:
        prov = sr.get("provenance")
        if not prov:
            continue
        url = sr["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        rows.append(
            {
                "url": url,
                "title": sr.get("title", "")[:60],
                "kind": prov.get("source_kind", "unknown"),
                "extractor": prov.get("extractor", "unknown"),
                "hash": prov.get("content_hash", "")[:8],
            }
        )

    if not rows:
        return ""

    lines: list[str] = []
    lines.append("| Source | Kind | Extractor | Hash |")
    lines.append("|--------|------|-----------|------|")

    for row in rows:
        title = row["title"] or row["url"][:50]
        lines.append(
            f"| [{title}]({row['url']}) | {row['kind']} | {row['extractor']} | `{row['hash']}` |"
        )

    return "\n".join(lines)

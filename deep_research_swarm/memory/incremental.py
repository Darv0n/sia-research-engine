"""Incremental research — content-hash-based diff for re-runs (V8, PR-10, OE2).

When re-running research on a topic, skip re-extraction of unchanged sources.
Uses ProvenanceRecord.content_hash to detect which sources have changed.

All functions are deterministic and LLM-free.
"""

from __future__ import annotations

from deep_research_swarm.contracts import ExtractedContent, ProvenanceRecord


def compute_content_diff(
    current_provenance: list[ProvenanceRecord],
    previous_provenance: list[ProvenanceRecord],
) -> dict[str, str]:
    """Compare provenance records to identify changed, new, and unchanged sources.

    Returns a dict of {source_url: status} where status is one of:
    - "new": source URL not seen in previous run
    - "changed": source URL exists but content_hash differs
    - "unchanged": source URL and content_hash match
    """
    previous_by_url: dict[str, str] = {}
    for prov in previous_provenance:
        url = prov["source_url"]
        content_hash = prov.get("content_hash", "")
        if url and content_hash:
            previous_by_url[url] = content_hash

    diff: dict[str, str] = {}
    for prov in current_provenance:
        url = prov["source_url"]
        current_hash = prov.get("content_hash", "")

        if url not in previous_by_url:
            diff[url] = "new"
        elif not current_hash or not previous_by_url[url]:
            # Can't compare without hashes — treat as new
            diff[url] = "new"
        elif current_hash == previous_by_url[url]:
            diff[url] = "unchanged"
        else:
            diff[url] = "changed"

    return diff


def filter_unchanged_sources(
    extracted_contents: list[ExtractedContent],
    diff: dict[str, str],
) -> tuple[list[ExtractedContent], list[ExtractedContent]]:
    """Partition extracted contents into changed and unchanged.

    Returns (to_process, skipped) where:
    - to_process: new or changed sources that need re-extraction
    - skipped: unchanged sources that can reuse previous results
    """
    to_process: list[ExtractedContent] = []
    skipped: list[ExtractedContent] = []

    for ec in extracted_contents:
        url = ec["url"]
        status = diff.get(url, "new")
        if status == "unchanged":
            skipped.append(ec)
        else:
            to_process.append(ec)

    return to_process, skipped


def compute_diff_summary(diff: dict[str, str]) -> dict[str, int]:
    """Summarize a content diff into counts by status.

    Returns {"new": N, "changed": N, "unchanged": N}.
    """
    summary: dict[str, int] = {"new": 0, "changed": 0, "unchanged": 0}
    for status in diff.values():
        summary[status] = summary.get(status, 0) + 1
    return summary

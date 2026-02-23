"""Provenance assembly and tracking (V7, PR-06).

Builds ProvenanceRecord objects from extraction context and
attaches them to SearchResults during the extraction phase.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from deep_research_swarm.contracts import ProvenanceRecord, SearchResult


def build_provenance(
    *,
    url: str,
    extractor: str,
    content: str,
    source_kind: str = "web",
    capture_timestamp: str = "",
) -> ProvenanceRecord:
    """Build a ProvenanceRecord from extraction context.

    Args:
        url: The source URL.
        extractor: Name of the extractor used.
        content: Extracted text content (for content hash).
        source_kind: "web", "scholarly", "archive", "pdf".
        capture_timestamp: Wayback-only, YYYYMMDDHHMMSS format.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16] if content else ""

    return ProvenanceRecord(
        entity_id=f"urn:url:{url}",
        source_url=url,
        source_kind=source_kind,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        extractor=extractor,
        license_tag="unknown",
        capture_timestamp=capture_timestamp,
        content_hash=content_hash,
    )


def attach_provenance_to_result(
    result: SearchResult,
    *,
    extractor: str,
    content: str,
) -> SearchResult:
    """Attach or update provenance on a SearchResult after extraction.

    If the result already has provenance (e.g., from a scholarly backend),
    only updates the content_hash. Otherwise creates a new ProvenanceRecord.
    """
    existing = result.get("provenance")

    if existing:
        # Update content_hash on existing provenance
        updated = {**existing, "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16]}
        return {**result, "provenance": updated}

    # Build new provenance
    source_kind = "web"
    if "wayback" in extractor:
        source_kind = "archive"
    elif extractor == "pdf" or extractor == "pymupdf4llm":
        source_kind = "pdf"

    provenance = build_provenance(
        url=result["url"],
        extractor=extractor,
        content=content,
        source_kind=source_kind,
    )
    return {**result, "provenance": provenance}

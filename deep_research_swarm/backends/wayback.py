"""Wayback Machine archive backend (V7, PR-05).

Fallback-only, not primary search (D6). Called when live extraction
fails for a URL. Uses CDX API for snapshot discovery and id_ replay
URLs for content retrieval.

Polite 1-second spacing between requests.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone

import httpx

from deep_research_swarm.contracts import (
    ArchiveCapture,
    ProvenanceRecord,
    SearchResult,
)
from deep_research_swarm.scoring.authority import classify_authority

from . import register_backend

_CDX_BASE = "https://web.archive.org/cdx/search/cdx"
_WAYBACK_BASE = "https://web.archive.org"
_DEFAULT_TIMEOUT = 15.0
_POLITE_DELAY = 1.0  # seconds between requests


class WaybackBackend:
    """Wayback Machine archive backend.

    search() returns empty for non-URL queries (D6 — fallback only).
    Use lookup_url(), get_captures(), and fetch_archived_content()
    directly for archive operations.
    """

    name: str = "wayback"

    def __init__(self, *, timeout: int = 15) -> None:
        self._timeout = float(timeout)
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "deep-research-swarm/0.7"},
            follow_redirects=True,
        )
        self._last_request_time: float = 0.0

    async def _polite_delay(self) -> None:
        """Enforce polite 1-second spacing between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < _POLITE_DELAY:
            await asyncio.sleep(_POLITE_DELAY - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Search returns empty for non-URL queries (D6).

        Wayback has no full-text search. Only processes URL-like queries
        by looking up archived snapshots.
        """
        # Only process URL-like queries
        if not re.match(r"https?://", query):
            return []

        captures = await self.get_captures(query, limit=num_results)
        if not captures:
            return []

        results: list[SearchResult] = []
        now = datetime.now(timezone.utc).isoformat()

        for rank, capture in enumerate(captures[:num_results], start=1):
            archive_url = capture["archive_url"]
            original_url = capture["original_url"]

            provenance = ProvenanceRecord(
                entity_id=f"urn:wayback:{capture['capture_timestamp']}:{original_url}",
                source_url=archive_url,
                source_kind="archive",
                fetched_at=now,
                extractor="wayback",
                license_tag="unknown",
                capture_timestamp=capture["capture_timestamp"],
                content_hash="",
            )

            results.append(
                SearchResult(
                    id=f"sr-{uuid.uuid4().hex[:8]}",
                    sub_query_id="",
                    url=archive_url,
                    title=f"Archived: {original_url}",
                    snippet=f"Wayback snapshot from {capture['capture_timestamp'][:8]}",
                    backend=self.name,
                    rank=rank,
                    score=round(1.0 / (60 + rank), 6),
                    authority=classify_authority(original_url),
                    timestamp=now,
                    provenance=provenance,
                )
            )

        return results

    async def lookup_url(self, url: str) -> ArchiveCapture | None:
        """Look up the most recent Wayback snapshot for a URL.

        Returns the newest available capture, or None if not archived.
        """
        captures = await self.get_captures(url, limit=1)
        return captures[0] if captures else None

    async def get_captures(self, url: str, *, limit: int = 5) -> list[ArchiveCapture]:
        """Query CDX API for available snapshots of a URL.

        Returns captures sorted by timestamp descending (newest first).
        """
        await self._polite_delay()

        params = {
            "url": url,
            "output": "json",
            "limit": str(limit),
            "fl": "timestamp,original,statuscode,mimetype",
            "sort": "reverse",  # Newest first
            "filter": "statuscode:200",
        }

        try:
            resp = await self._client.get(_CDX_BASE, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return []

        # CDX returns header row + data rows
        if len(data) < 2:
            return []

        captures: list[ArchiveCapture] = []
        for row in data[1:]:
            if len(row) < 4:
                continue
            timestamp, original, status_code, mimetype = row[0], row[1], row[2], row[3]
            # Use id_ replay URL to get original content without Wayback toolbar
            archive_url = f"{_WAYBACK_BASE}/web/{timestamp}id_/{original}"
            captures.append(
                ArchiveCapture(
                    original_url=original,
                    archive_url=archive_url,
                    capture_timestamp=timestamp,
                    status_code=int(status_code) if status_code.isdigit() else 200,
                    content_type=mimetype,
                )
            )

        return captures

    async def fetch_archived_content(self, capture: ArchiveCapture) -> str:
        """Fetch the content of an archived page.

        Uses id_ replay URL to get clean content without Wayback toolbar.
        """
        await self._polite_delay()

        try:
            resp = await self._client.get(capture["archive_url"])
            if resp.status_code != 200:
                return ""
            return resp.text
        except httpx.HTTPError:
            return ""

    async def health_check(self) -> bool:
        """Check that Wayback CDX API is reachable."""
        try:
            resp = await self._client.get(
                _CDX_BASE,
                params={"url": "example.com", "output": "json", "limit": "1"},
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


register_backend("wayback", WaybackBackend)

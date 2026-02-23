"""OpenAlex scholarly search backend (V7, PR-02).

Uses the OpenAlex REST API to search academic papers.
Populates scholarly_metadata + provenance on every SearchResult.
Reconstructs abstracts from OpenAlex inverted index format.
429 retry with exponential backoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone

import httpx

from deep_research_swarm.contracts import (
    ProvenanceRecord,
    ScholarlyMetadata,
    SearchResult,
    SourceAuthority,
)

from . import register_backend

_BASE_URL = "https://api.openalex.org"
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format.

    OpenAlex stores abstracts as {word: [position_indices]} mappings.
    Reconstruct by placing each word at its position(s).
    """
    if not inverted_index:
        return ""

    # Find max position to size the array
    max_pos = 0
    for positions in inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))

    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            if pos <= max_pos:
                words[pos] = word

    return " ".join(w for w in words if w)


def _extract_authors(authorships: list[dict]) -> list[str]:
    """Extract author display names from OpenAlex authorships."""
    authors: list[str] = []
    for authorship in authorships:
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)
    return authors


def _build_scholarly_metadata(work: dict) -> ScholarlyMetadata:
    """Build ScholarlyMetadata from an OpenAlex work object."""
    ids = work.get("ids", {})
    doi_raw = ids.get("doi", "") or work.get("doi", "") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""

    # Open access info
    oa = work.get("open_access", {})
    oa_url = oa.get("oa_url", "") or ""

    # Venue / source
    primary_location = work.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}
    venue = source.get("display_name", "") or ""

    return ScholarlyMetadata(
        doi=doi,
        arxiv_id="",  # OpenAlex doesn't reliably surface arXiv IDs
        pmid=ids.get("pmid", "") or "",
        title=work.get("title", "") or "",
        authors=_extract_authors(work.get("authorships", [])),
        year=work.get("publication_year", 0) or 0,
        venue=venue,
        citation_count=work.get("cited_by_count", 0) or 0,
        reference_count=len(work.get("referenced_works", [])),
        is_open_access=oa.get("is_oa", False) or False,
        open_access_url=oa_url,
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
    )


def _build_provenance(work: dict, fetched_at: str) -> ProvenanceRecord:
    """Build ProvenanceRecord from an OpenAlex work object."""
    doi_raw = work.get("doi", "") or ""
    url = doi_raw or work.get("id", "") or ""
    title = work.get("title", "") or ""

    return ProvenanceRecord(
        entity_id=f"urn:openalex:{work.get('id', '')}",
        source_url=url,
        source_kind="scholarly",
        fetched_at=fetched_at,
        extractor="openalex",
        license_tag="CC0",  # OpenAlex data is CC0
        capture_timestamp="",
        content_hash=hashlib.sha256(title.encode()).hexdigest()[:16],
    )


class OpenAlexBackend:
    """OpenAlex scholarly search backend.

    Requires email for polite pool access (higher rate limits).
    API key is optional and provides even higher limits.
    """

    name: str = "openalex"

    def __init__(
        self,
        *,
        email: str = "",
        api_key: str = "",
    ) -> None:
        self._email = email
        self._api_key = api_key
        headers: dict[str, str] = {"User-Agent": "deep-research-swarm/0.7"}
        if email:
            headers["From"] = email
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    async def _request_with_retry(self, path: str, params: dict) -> httpx.Response | None:
        """Make a GET request with exponential backoff on 429."""
        backoff = _INITIAL_BACKOFF
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.get(path, params=params)
                if resp.status_code == 429:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError:
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                return None
            except httpx.HTTPError:
                return None
        return None

    async def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Search OpenAlex works by query string."""
        params: dict[str, str | int] = {
            "search": query,
            "per_page": min(num_results, 50),
        }
        if self._email:
            params["mailto"] = self._email

        resp = await self._request_with_retry("/works", params)
        if resp is None:
            return []

        try:
            data = resp.json()
        except ValueError:
            return []

        results: list[SearchResult] = []
        now = datetime.now(timezone.utc).isoformat()
        works = data.get("results", [])

        for rank, work in enumerate(works[:num_results], start=1):
            doi_raw = work.get("doi", "") or ""
            url = doi_raw or work.get("id", "") or ""
            if not url:
                continue

            title = work.get("title", "") or ""
            scholarly = _build_scholarly_metadata(work)
            snippet = scholarly["abstract"][:300] if scholarly["abstract"] else title

            results.append(
                SearchResult(
                    id=f"sr-{uuid.uuid4().hex[:8]}",
                    sub_query_id="",  # Set by searcher agent
                    url=url,
                    title=title,
                    snippet=snippet,
                    backend=self.name,
                    rank=rank,
                    score=round(1.0 / (60 + rank), 6),
                    authority=SourceAuthority.INSTITUTIONAL,
                    timestamp=now,
                    provenance=_build_provenance(work, now),
                    scholarly_metadata=scholarly,
                )
            )

        return results

    async def health_check(self) -> bool:
        """Check that OpenAlex API is reachable."""
        try:
            resp = await self._client.get("/works", params={"search": "test", "per_page": 1})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


register_backend("openalex", OpenAlexBackend)

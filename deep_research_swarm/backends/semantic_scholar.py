"""Semantic Scholar search backend (V7, PR-03).

Uses the Semantic Scholar Academic Graph API to search papers.
Includes get_paper_details() for citation chaining (PR-08).
Handles Retry-After header for rate limiting.
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

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0

# Fields we request from the S2 API
_SEARCH_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,citationCount,"
    "referenceCount,isOpenAccess,openAccessPdf,authors,url"
)
_DETAIL_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,citationCount,"
    "referenceCount,isOpenAccess,openAccessPdf,authors,url,"
    "references.paperId,references.title,citations.paperId,citations.title"
)


def _build_scholarly_metadata(paper: dict) -> ScholarlyMetadata:
    """Build ScholarlyMetadata from an S2 paper object."""
    ext_ids = paper.get("externalIds", {}) or {}
    oa_pdf = paper.get("openAccessPdf", {}) or {}

    authors = []
    for a in paper.get("authors", []):
        name = a.get("name", "")
        if name:
            authors.append(name)

    return ScholarlyMetadata(
        doi=ext_ids.get("DOI", "") or "",
        arxiv_id=ext_ids.get("ArXiv", "") or "",
        pmid=ext_ids.get("PubMed", "") or "",
        title=paper.get("title", "") or "",
        authors=authors,
        year=paper.get("year", 0) or 0,
        venue=paper.get("venue", "") or "",
        citation_count=paper.get("citationCount", 0) or 0,
        reference_count=paper.get("referenceCount", 0) or 0,
        is_open_access=paper.get("isOpenAccess", False) or False,
        open_access_url=oa_pdf.get("url", "") or "",
        abstract=paper.get("abstract", "") or "",
    )


def _build_provenance(paper: dict, fetched_at: str) -> ProvenanceRecord:
    """Build ProvenanceRecord from an S2 paper object."""
    title = paper.get("title", "") or ""
    paper_id = paper.get("paperId", "") or ""
    url = paper.get("url", "") or f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

    return ProvenanceRecord(
        entity_id=f"urn:s2:{paper_id}",
        source_url=url,
        source_kind="scholarly",
        fetched_at=fetched_at,
        extractor="semantic_scholar",
        license_tag="unknown",
        capture_timestamp="",
        content_hash=hashlib.sha256(title.encode()).hexdigest()[:16],
    )


def _paper_url(paper: dict) -> str:
    """Get the best URL for a paper."""
    url = paper.get("url", "")
    if url:
        return url
    ext_ids = paper.get("externalIds", {}) or {}
    doi = ext_ids.get("DOI", "")
    if doi:
        return f"https://doi.org/{doi}"
    arxiv = ext_ids.get("ArXiv", "")
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv}"
    paper_id = paper.get("paperId", "")
    return f"https://www.semanticscholar.org/paper/{paper_id}"


class SemanticScholarBackend:
    """Semantic Scholar search backend.

    Works without authentication (lower rate limits).
    API key optional for higher limits.
    """

    name: str = "semantic_scholar"

    def __init__(self, *, api_key: str = "") -> None:
        self._api_key = api_key
        headers: dict[str, str] = {"User-Agent": "deep-research-swarm/0.7"}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    async def _request_with_retry(self, path: str, params: dict) -> httpx.Response | None:
        """Make a GET request with Retry-After handling."""
        backoff = _INITIAL_BACKOFF
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.get(path, params=params)
                if resp.status_code == 429:
                    # Respect Retry-After header if present
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else backoff
                    await asyncio.sleep(min(wait, 30.0))
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
        """Search Semantic Scholar papers by query string."""
        params: dict[str, str | int] = {
            "query": query,
            "limit": min(num_results, 100),
            "fields": _SEARCH_FIELDS,
        }

        resp = await self._request_with_retry("/paper/search", params)
        if resp is None:
            return []

        try:
            data = resp.json()
        except ValueError:
            return []

        results: list[SearchResult] = []
        now = datetime.now(timezone.utc).isoformat()
        papers = data.get("data", [])

        for rank, paper in enumerate(papers[:num_results], start=1):
            url = _paper_url(paper)
            if not url:
                continue

            title = paper.get("title", "") or ""
            scholarly = _build_scholarly_metadata(paper)
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
                    provenance=_build_provenance(paper, now),
                    scholarly_metadata=scholarly,
                )
            )

        return results

    async def get_paper_details(self, paper_id: str) -> dict | None:
        """Fetch detailed paper info including references and citations.

        Used by citation chaining (PR-08). Returns raw S2 paper dict
        or None on failure.
        """
        params: dict[str, str] = {"fields": _DETAIL_FIELDS}
        resp = await self._request_with_retry(f"/paper/{paper_id}", params)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    async def health_check(self) -> bool:
        """Check that Semantic Scholar API is reachable."""
        try:
            resp = await self._client.get("/paper/search", params={"query": "test", "limit": 1})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


register_backend("semantic_scholar", SemanticScholarBackend)

"""SearXNG search backend."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx

from deep_research_swarm.contracts import SearchResult
from deep_research_swarm.scoring.authority import classify_authority

from . import register_backend


class SearXNGBackend:
    name: str = "searxng"

    def __init__(self, *, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        category: str | None = None,
    ) -> list[SearchResult]:
        params: dict = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if category:
            params["categories"] = category

        try:
            resp = await self._client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return []

        results: list[SearchResult] = []
        now = datetime.now(timezone.utc).isoformat()

        for rank, item in enumerate(data.get("results", [])[:num_results], start=1):
            url = item.get("url", "")
            results.append(
                SearchResult(
                    id=f"sr-{uuid.uuid4().hex[:8]}",
                    sub_query_id="",  # Set by searcher agent
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    backend=self.name,
                    rank=rank,
                    score=round(1.0 / (60 + rank), 6),  # Initial RRF-like score
                    authority=classify_authority(url),
                    timestamp=now,
                )
            )

        return results

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(
                f"{self.base_url}/search",
                params={"q": "test", "format": "json"},
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


register_backend("searxng", SearXNGBackend)

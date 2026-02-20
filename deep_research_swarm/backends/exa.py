"""Exa semantic search backend (V2)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from deep_research_swarm.contracts import SearchResult
from deep_research_swarm.scoring.authority import classify_authority

from . import register_backend


class ExaBackend:
    name: str = "exa"

    def __init__(self, *, api_key: str = "") -> None:
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from exa_py import Exa

            self._client = Exa(api_key=self.api_key)
        return self._client

    async def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        category: str | None = None,
    ) -> list[SearchResult]:
        try:
            client = self._get_client()
            response = client.search(query, num_results=num_results, type="neural")
        except Exception:
            return []

        results: list[SearchResult] = []
        now = datetime.now(timezone.utc).isoformat()

        for rank, item in enumerate(response.results[:num_results], start=1):
            url = item.url or ""
            results.append(
                SearchResult(
                    id=f"sr-{uuid.uuid4().hex[:8]}",
                    sub_query_id="",
                    url=url,
                    title=item.title or "",
                    snippet=getattr(item, "text", "") or "",
                    backend=self.name,
                    rank=rank,
                    score=round(getattr(item, "score", 1.0 / (60 + rank)), 6),
                    authority=classify_authority(url),
                    timestamp=now,
                )
            )

        return results

    async def health_check(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False


register_backend("exa", ExaBackend)

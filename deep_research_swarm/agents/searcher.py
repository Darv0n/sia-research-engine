"""Searcher agent — dispatches sub-queries to search backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deep_research_swarm.backends import get_backend
from deep_research_swarm.contracts import SearchResult, SubQuery

if TYPE_CHECKING:
    from deep_research_swarm.backends.cache import SearchCache


async def search_sub_query(
    sub_query: SubQuery,
    *,
    backend_configs: dict[str, dict] | None = None,
    cache: SearchCache | None = None,
) -> list[SearchResult]:
    """Search a single sub-query across its assigned backends.

    Returns all results with sub_query_id populated.
    When a cache is provided, checks it before calling the backend and stores
    results after a successful backend call.
    """
    backend_configs = backend_configs or {}
    all_results: list[SearchResult] = []
    num_results = 10

    for backend_name in sub_query["search_backends"]:
        try:
            # Check cache first
            if cache is not None:
                cached = cache.get(sub_query["question"], backend_name, num_results)
                if cached is not None:
                    for r in cached:
                        r["sub_query_id"] = sub_query["id"]
                    all_results.extend(cached)
                    continue

            kwargs = backend_configs.get(backend_name, {})
            backend = get_backend(backend_name, **kwargs)
            results = await backend.search(sub_query["question"], num_results=num_results)

            # Store in cache before stamping sub_query_id (keep results generic)
            if cache is not None:
                cache.put(sub_query["question"], backend_name, num_results, results)

            # Stamp the sub_query_id on each result
            for r in results:
                r["sub_query_id"] = sub_query["id"]

            all_results.extend(results)
        except (KeyError, Exception):
            # Backend not available or failed — continue with others
            continue

    return all_results

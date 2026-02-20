"""Searcher agent — dispatches sub-queries to search backends."""

from __future__ import annotations

from deep_research_swarm.backends import get_backend
from deep_research_swarm.contracts import SearchResult, SubQuery


async def search_sub_query(
    sub_query: SubQuery,
    *,
    backend_configs: dict[str, dict] | None = None,
) -> list[SearchResult]:
    """Search a single sub-query across its assigned backends.

    Returns all results with sub_query_id populated.
    """
    backend_configs = backend_configs or {}
    all_results: list[SearchResult] = []

    for backend_name in sub_query["search_backends"]:
        try:
            kwargs = backend_configs.get(backend_name, {})
            backend = get_backend(backend_name, **kwargs)
            results = await backend.search(sub_query["question"], num_results=10)

            # Stamp the sub_query_id on each result
            for r in results:
                r["sub_query_id"] = sub_query["id"]

            all_results.extend(results)
        except (KeyError, Exception):
            # Backend not available or failed — continue with others
            continue

    return all_results

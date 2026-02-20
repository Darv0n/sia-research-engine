"""StateGraph construction — wires all nodes, edges, and fan-out patterns."""

from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.agents.contradiction import detect_contradictions
from deep_research_swarm.agents.critic import critique
from deep_research_swarm.agents.extractor import extract_content
from deep_research_swarm.agents.planner import plan
from deep_research_swarm.agents.searcher import search_sub_query
from deep_research_swarm.agents.synthesizer import synthesize
from deep_research_swarm.backends.cache import SearchCache
from deep_research_swarm.config import Settings
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.reporting.renderer import render_report
from deep_research_swarm.scoring.diversity import compute_diversity
from deep_research_swarm.scoring.rrf import build_scored_documents


def _get_stream_writer(config: RunnableConfig | None) -> callable | None:
    """Safely extract a stream writer from LangGraph config, if available."""
    if config is None:
        return None
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (ImportError, Exception):
        return None


def build_graph(
    settings: Settings,
    *,
    enable_cache: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the research graph.

    Returns a compiled StateGraph ready to invoke.
    """
    # Create agent callers with appropriate models
    opus_caller = AgentCaller(
        api_key=settings.anthropic_api_key,
        model=settings.opus_model,
        max_concurrent=settings.max_concurrent_requests,
    )
    sonnet_caller = AgentCaller(
        api_key=settings.anthropic_api_key,
        model=settings.sonnet_model,
        max_concurrent=settings.max_concurrent_requests,
    )

    # Search cache (opt-in via enable_cache flag)
    search_cache: SearchCache | None = None
    if enable_cache:
        search_cache = SearchCache(
            cache_dir=settings.search_cache_dir,
            ttl=settings.search_cache_ttl,
        )

    # Backend configs for searcher
    backend_configs: dict[str, dict] = {
        "searxng": {"base_url": settings.searxng_url},
    }
    if settings.exa_api_key:
        backend_configs["exa"] = {"api_key": settings.exa_api_key}
    if settings.tavily_api_key:
        backend_configs["tavily"] = {"api_key": settings.tavily_api_key}

    # --- Node functions (closures over callers) ---

    async def health_check_node(state: ResearchState) -> dict:
        """Verify search backends are reachable before dispatching."""
        import deep_research_swarm.backends.searxng  # noqa: F401

        if settings.exa_api_key:
            import deep_research_swarm.backends.exa  # noqa: F401
        if settings.tavily_api_key:
            import deep_research_swarm.backends.tavily  # noqa: F401

        from deep_research_swarm.backends import get_backend

        requested = state.get("search_backends", ["searxng"])
        healthy: list[str] = []

        for name in requested:
            try:
                kwargs = backend_configs.get(name, {})
                backend = get_backend(name, **kwargs)
                if await backend.health_check():
                    healthy.append(name)
            except Exception:
                pass

        if not healthy:
            raise RuntimeError(
                f"No healthy backends. Requested: {requested}. "
                "Check that SearXNG is running and API keys are valid."
            )

        return {"search_backends": healthy}

    async def plan_node(state: ResearchState) -> dict:
        return await plan(state, opus_caller)

    async def search_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
        """Fan-out: search all sub-queries from the current iteration in parallel."""
        writer = _get_stream_writer(config)
        sub_queries = state.get("sub_queries", [])
        if not sub_queries:
            return {"search_results": []}

        # Get the sub-queries added in the latest planning step
        # (all sub_queries accumulate via reducer, take the latest batch)
        prev_history = state.get("iteration_history", [])
        prev_sq_count = sum(h["sub_queries_generated"] for h in prev_history)
        latest_queries = sub_queries[prev_sq_count:]

        if not latest_queries:
            latest_queries = sub_queries

        # Import backends to trigger registration
        import deep_research_swarm.backends.searxng  # noqa: F401

        if settings.exa_api_key:
            import deep_research_swarm.backends.exa  # noqa: F401
        if settings.tavily_api_key:
            import deep_research_swarm.backends.tavily  # noqa: F401

        if writer:
            msg = f"dispatching {len(latest_queries)} queries"
            writer({"kind": "search_progress", "message": msg})

        tasks = [
            search_sub_query(sq, backend_configs=backend_configs, cache=search_cache)
            for sq in latest_queries
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for result in results_lists:
            if isinstance(result, list):
                all_results.extend(result)

        if writer:
            writer({"kind": "search_progress", "message": "complete", "count": len(all_results)})

        return {"search_results": all_results}

    async def extract_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
        """Fan-out: extract content from all search results in parallel."""
        writer = _get_stream_writer(config)
        search_results = state.get("search_results", [])
        if not search_results:
            return {"extracted_contents": []}

        # Deduplicate by URL, take latest results
        seen_urls: set[str] = set()
        unique_results = []
        for sr in reversed(search_results):
            if sr["url"] not in seen_urls:
                seen_urls.add(sr["url"])
                unique_results.append(sr)

        capped = unique_results[:30]

        if writer:
            writer({"kind": "extract_progress", "message": f"extracting {len(capped)} URLs"})

        # Limit concurrent extractions
        sem = asyncio.Semaphore(settings.max_concurrent_requests)

        async def extract_with_sem(sr):
            async with sem:
                return await extract_content(sr)

        tasks = [extract_with_sem(sr) for sr in capped]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extracted = []
        for result in results:
            if isinstance(result, dict):
                extracted.append(result)

        if writer:
            writer({"kind": "extract_progress", "message": "complete", "count": len(extracted)})

        return {"extracted_contents": extracted}

    async def score_node(state: ResearchState) -> dict:
        """Score and rank all documents using RRF + authority, compute diversity."""
        search_results = state.get("search_results", [])
        extracted_contents = state.get("extracted_contents", [])

        scored = build_scored_documents(
            search_results,
            extracted_contents,
            k=settings.rrf_k,
            authority_weight=settings.authority_weight,
        )

        diversity = compute_diversity(scored)

        return {"scored_documents": scored, "diversity_metrics": diversity}

    async def contradiction_node(state: ResearchState) -> dict:
        """Detect contradictions among scored documents."""
        scored_docs = state.get("scored_documents", [])
        if len(scored_docs) < 2:
            return {"contradictions": []}

        contradictions, usage = await detect_contradictions(scored_docs, sonnet_caller)
        result: dict = {"contradictions": contradictions}
        if usage:
            result["token_usage"] = [usage]
        return result

    async def synthesize_node(state: ResearchState) -> dict:
        return await synthesize(state, opus_caller)

    async def critique_node(state: ResearchState) -> dict:
        return await critique(state, sonnet_caller)

    async def rollup_budget_node(state: ResearchState) -> dict:
        """Roll up token_usage list into totals for budget tracking."""
        usage_list = state.get("token_usage", [])
        total_tokens = sum(u["input_tokens"] + u["output_tokens"] for u in usage_list)
        total_cost = sum(u["cost_usd"] for u in usage_list)
        return {
            "total_tokens_used": total_tokens,
            "total_cost_usd": round(total_cost, 6),
        }

    async def report_node(state: ResearchState) -> dict:
        """Generate final report from section drafts."""
        report = render_report(state)
        return {"final_report": report}

    # --- Routing ---

    def should_continue(state: ResearchState) -> str:
        """Route after budget rollup: continue or finish."""
        if state.get("converged", False):
            return "report"
        return "plan"

    # --- Build graph ---

    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("health_check", health_check_node)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("extract", extract_node)
    graph.add_node("score", score_node)
    graph.add_node("contradiction", contradiction_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critique", critique_node)
    graph.add_node("rollup_budget", rollup_budget_node)
    graph.add_node("report", report_node)

    # Wire edges: health_check -> plan -> ... -> score -> contradiction -> synthesize -> ...
    graph.set_entry_point("health_check")
    graph.add_edge("health_check", "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "score")
    graph.add_edge("score", "contradiction")
    graph.add_edge("contradiction", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_edge("critique", "rollup_budget")

    # Conditional: rollup_budget -> report (converged) or rollup_budget -> plan (re-plan)
    graph.add_conditional_edges(
        "rollup_budget",
        should_continue,
        {"report": "report", "plan": "plan"},
    )

    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)

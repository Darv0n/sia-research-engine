"""StateGraph construction — wires all nodes, edges, and fan-out patterns."""

from __future__ import annotations

import asyncio
import inspect
import time
import warnings
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deep_research_swarm.adaptive.adapt_extraction import adapt_extraction_node
from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.agents.citation_chain import citation_chain
from deep_research_swarm.agents.contradiction import detect_contradictions
from deep_research_swarm.agents.critic import critique
from deep_research_swarm.agents.extractor import extract_content
from deep_research_swarm.agents.planner import plan
from deep_research_swarm.agents.searcher import search_sub_query
from deep_research_swarm.agents.synthesizer import synthesize
from deep_research_swarm.backends.cache import SearchCache
from deep_research_swarm.config import Settings
from deep_research_swarm.extractors.chunker import chunk_all_documents
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.reporting.renderer import render_report
from deep_research_swarm.scoring.diversity import compute_diversity
from deep_research_swarm.scoring.rrf import build_scored_documents

if TYPE_CHECKING:
    from deep_research_swarm.event_log.writer import EventLog


def _get_stream_writer(config: RunnableConfig | None) -> callable | None:
    """Safely extract a stream writer from LangGraph config, if available."""
    if config is None:
        return None
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (ImportError, Exception):
        return None


def _summarize_dict(d: dict) -> dict[str, int]:
    """Produce a {field: count_or_len} summary for event logging."""
    summary: dict[str, int] = {}
    for key, val in d.items():
        if isinstance(val, list):
            summary[key] = len(val)
        elif isinstance(val, str):
            summary[key] = len(val)
        elif isinstance(val, (int, float)):
            summary[key] = 1
        elif isinstance(val, dict):
            summary[key] = len(val)
    return summary


def _wrap_with_logging(
    node_name: str,
    fn: callable,
    event_log: EventLog,
) -> callable:
    """Wrap a node function to emit a RunEvent after execution."""
    params = inspect.signature(fn).parameters
    has_config = len(params) >= 2

    if has_config:

        async def logged_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
            start = time.monotonic()
            inputs_summary = _summarize_dict(state)
            result = await fn(state, config)
            elapsed = time.monotonic() - start
            token_usage = result.get("token_usage", [])
            tokens = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in token_usage)
            cost = sum(u.get("cost_usd", 0.0) for u in token_usage)
            event = event_log.make_event(
                node=node_name,
                iteration=state.get("current_iteration", 0),
                elapsed_s=elapsed,
                inputs_summary=inputs_summary,
                outputs_summary=_summarize_dict(result),
                tokens=tokens,
                cost=cost,
            )
            event_log.emit(event)
            return result

    else:

        async def logged_node(state: ResearchState) -> dict:
            start = time.monotonic()
            inputs_summary = _summarize_dict(state)
            result = await fn(state)
            elapsed = time.monotonic() - start
            token_usage = result.get("token_usage", [])
            tokens = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in token_usage)
            cost = sum(u.get("cost_usd", 0.0) for u in token_usage)
            event = event_log.make_event(
                node=node_name,
                iteration=state.get("current_iteration", 0),
                elapsed_s=elapsed,
                inputs_summary=inputs_summary,
                outputs_summary=_summarize_dict(result),
                tokens=tokens,
                cost=cost,
            )
            event_log.emit(event)
            return result

    return logged_node


def build_graph(
    settings: Settings,
    *,
    enable_cache: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
    event_log: EventLog | None = None,
    mode: str = "auto",
) -> CompiledStateGraph:
    """Build and compile the research graph.

    Returns a compiled StateGraph ready to invoke.
    """
    # Create agent callers with appropriate models
    opus_caller = AgentCaller(
        api_key=settings.anthropic_api_key,
        model=settings.opus_model,
        max_concurrent=settings.max_concurrent_requests,
        fallback_model=settings.sonnet_model,
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

    # Semantic Scholar backend for citation chaining (PR-08)
    s2_backend = None
    try:
        from deep_research_swarm.backends.semantic_scholar import SemanticScholarBackend

        s2_backend = SemanticScholarBackend(api_key=settings.semantic_scholar_api_key)
    except Exception:
        pass

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
        return await plan(state, opus_caller, available_backends=settings.available_backends())

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

    async def chunk_passages_node(state: ResearchState) -> dict:
        """Chunk extracted documents into SourcePassage objects (V7, PR-10)."""
        extracted_contents = state.get("extracted_contents", [])
        scored_documents = state.get("scored_documents", [])

        # On first iteration, scored_documents may be empty.
        # Use search_results to build minimal scored docs for URL matching.
        if not scored_documents:
            search_results = state.get("search_results", [])
            # Build temporary scored docs from search results for chunking
            from deep_research_swarm.contracts import ScoredDocument, SourceAuthority

            temp_scored: dict[str, ScoredDocument] = {}
            for sr in search_results:
                url = sr["url"]
                if url not in temp_scored:
                    temp_scored[url] = ScoredDocument(
                        id=sr["id"],
                        url=url,
                        title=sr["title"],
                        content="",
                        rrf_score=0.0,
                        authority=sr.get("authority", SourceAuthority.UNKNOWN),
                        authority_score=0.0,
                        combined_score=0.0,
                        sub_query_ids=[sr["sub_query_id"]],
                    )
            scored_documents = list(temp_scored.values())

        passages = chunk_all_documents(extracted_contents, scored_documents)
        return {"source_passages": passages}

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

    async def citation_chain_node(state: ResearchState) -> dict:
        """Expand evidence via citation graph traversal (V7, PR-08)."""
        return await citation_chain(state, s2_backend)

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
        return await critique(
            state, sonnet_caller, convergence_threshold=settings.convergence_threshold
        )

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

    # --- Event log wrapping ---

    node_map: dict[str, callable] = {
        "health_check": health_check_node,
        "plan": plan_node,
        "search": search_node,
        "adapt_extraction": adapt_extraction_node,
        "extract": extract_node,
        "chunk_passages": chunk_passages_node,
        "score": score_node,
        "citation_chain": citation_chain_node,
        "contradiction": contradiction_node,
        "synthesize": synthesize_node,
        "critique": critique_node,
        "rollup_budget": rollup_budget_node,
        "report": report_node,
    }

    if event_log is not None:
        node_map = {name: _wrap_with_logging(name, fn, event_log) for name, fn in node_map.items()}

    # --- HITL gate nodes (only when mode=hitl + checkpointer exists) ---

    hitl_mode = mode == "hitl" and checkpointer is not None

    if hitl_mode:
        from langgraph.types import interrupt

        async def plan_gate_node(state: ResearchState) -> dict:
            """Interrupt after plan for human review of research plan."""
            interrupt(
                {
                    "gate": "plan_gate",
                    "iteration": state.get("current_iteration", 0),
                    "perspectives": state.get("perspectives", []),
                    "sub_queries": [sq["question"] for sq in state.get("sub_queries", [])],
                    "message": "Review the research plan. Resume to continue.",
                }
            )
            return {}

        async def report_gate_node(state: ResearchState) -> dict:
            """Interrupt after report for human review of final output."""
            report = state.get("final_report", "")
            interrupt(
                {
                    "gate": "report_gate",
                    "iterations": len(state.get("iteration_history", [])),
                    "cost": state.get("total_cost_usd", 0.0),
                    "report_preview": report[:500],
                    "report_length": len(report),
                    "message": "Review the final report. Resume to accept.",
                }
            )
            return {}

        node_map["plan_gate"] = plan_gate_node
        node_map["report_gate"] = report_gate_node

    # --- Build graph ---

    graph = StateGraph(ResearchState)

    # Add nodes (suppress LangGraph config-typing advisory)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The 'config' parameter should be typed as",
            category=UserWarning,
        )
        for name, fn in node_map.items():
            graph.add_node(name, fn)

    # Wire edges: health_check -> plan -> ... -> score -> contradiction -> synthesize -> ...
    graph.set_entry_point("health_check")
    graph.add_edge("health_check", "plan")

    if hitl_mode:
        graph.add_edge("plan", "plan_gate")
        graph.add_edge("plan_gate", "search")
    else:
        graph.add_edge("plan", "search")

    graph.add_edge("search", "adapt_extraction")
    graph.add_edge("adapt_extraction", "extract")
    graph.add_edge("extract", "chunk_passages")
    graph.add_edge("chunk_passages", "score")
    graph.add_edge("score", "citation_chain")
    graph.add_edge("citation_chain", "contradiction")
    graph.add_edge("contradiction", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_edge("critique", "rollup_budget")

    # Conditional: rollup_budget -> report (converged) or rollup_budget -> plan (re-plan)
    graph.add_conditional_edges(
        "rollup_budget",
        should_continue,
        {"report": "report", "plan": "plan"},
    )

    if hitl_mode:
        graph.add_edge("report", "report_gate")
        graph.add_edge("report_gate", END)
    else:
        graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)

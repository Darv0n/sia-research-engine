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
from deep_research_swarm.adaptive.adapt_synthesis import adapt_synthesis_node
from deep_research_swarm.agents.base import AgentCaller
from deep_research_swarm.agents.citation_chain import citation_chain
from deep_research_swarm.agents.clarifier import clarify
from deep_research_swarm.agents.contradiction import detect_contradictions
from deep_research_swarm.agents.critic import critique
from deep_research_swarm.agents.extractor import extract_content
from deep_research_swarm.agents.gap_analyzer import analyze_gaps
from deep_research_swarm.agents.planner import plan
from deep_research_swarm.agents.searcher import search_sub_query
from deep_research_swarm.agents.synthesizer import synthesize
from deep_research_swarm.backends.cache import SearchCache
from deep_research_swarm.compress.artifact import build_knowledge_artifact
from deep_research_swarm.config import Settings
from deep_research_swarm.deliberate.merge import merge_judgments
from deep_research_swarm.deliberate.panel import (
    authority_judge,
    contradiction_judge,
    coverage_judge,
    grounding_judge,
)
from deep_research_swarm.extractors.chunker import chunk_all_documents
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.reporting.renderer import render_report
from deep_research_swarm.scoring.diversity import compute_diversity
from deep_research_swarm.scoring.rrf import build_scored_documents
from deep_research_swarm.sia.entropy import (
    compute_entropy,
    detect_dominance,
    detect_false_convergence,
    entropy_gate,
)
from deep_research_swarm.sia.singularity_prevention import singularity_check

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
    has_config = "config" in params

    is_async = asyncio.iscoroutinefunction(fn)

    if has_config:

        async def logged_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
            start = time.monotonic()
            inputs_summary = _summarize_dict(state)
            result = (await fn(state, config)) if is_async else fn(state, config)
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
            result = (await fn(state)) if is_async else fn(state)
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
    haiku_caller = AgentCaller(
        api_key=settings.anthropic_api_key,
        model=settings.haiku_model,
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

    async def clarify_node(state: ResearchState) -> dict:
        """V9: Pre-research scope analysis (G5). Auto mode = heuristic, no LLM."""
        resolved_mode = mode
        return await clarify(
            state,
            sonnet_caller if resolved_mode == "hitl" else None,
            mode=resolved_mode,
        )

    async def plan_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
        result = await plan(state, opus_caller, available_backends=settings.available_backends())

        # V9: Reset follow-up round for new iteration
        result["follow_up_round"] = 0
        # Reset convergence flags for new iteration (prevents stale state on resume)
        result["converged"] = False
        result["convergence_reason"] = ""

        # V9: Stream plan in all modes for transparency (G6)
        writer = _get_stream_writer(config)
        if writer:
            perspectives = result.get("perspectives", [])
            sub_queries = result.get("sub_queries", [])
            writer(
                {
                    "kind": "plan_summary",
                    "perspectives": perspectives,
                    "queries": [sq["question"] for sq in sub_queries],
                    "iteration": result.get("current_iteration", 1),
                }
            )

        return result

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

        # Read adaptive tunable (V8) — fall back to 10
        _snap = state.get("tunable_snapshot", {})
        num_results = int(_snap.get("results_per_query", 10))

        tasks = [
            search_sub_query(
                sq,
                backend_configs=backend_configs,
                cache=search_cache,
                num_results=num_results,
            )
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

        # V9: Prioritize URLs by score before capping (G3 — source volume)
        unique_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

        # Read adaptive tunables (V8) — fall back to V9 defaults
        _snap = state.get("tunable_snapshot", {})
        extraction_cap = int(_snap.get("extraction_cap", 50))
        content_trunc = int(_snap.get("content_truncation_chars", 50000))

        capped = unique_results[:extraction_cap]

        if writer:
            writer({"kind": "extract_progress", "message": f"extracting {len(capped)} URLs"})

        # Limit concurrent extractions
        sem = asyncio.Semaphore(settings.max_concurrent_requests)

        _grobid_url = settings.grobid_url

        async def extract_with_sem(sr):
            async with sem:
                return await extract_content(
                    sr, content_truncation_chars=content_trunc, grobid_url=_grobid_url
                )

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

    async def score_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
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

        # V9: Rich streaming — emit top sources preview
        writer = _get_stream_writer(config)
        if writer and scored:
            top = sorted(scored, key=lambda d: d["combined_score"], reverse=True)[:5]
            writer(
                {
                    "kind": "findings_preview",
                    "top_sources": [f"{d['title']} ({d['url']})" for d in top],
                }
            )

        return {"scored_documents": scored, "diversity_metrics": diversity}

    async def gap_analysis_node(state: ResearchState) -> dict:
        """V9: Analyze scored docs to identify knowledge gaps and generate follow-ups."""
        return await analyze_gaps(
            state, sonnet_caller, available_backends=settings.available_backends()
        )

    async def search_followup_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """V9: Execute follow-up queries identified by gap analysis."""
        writer = _get_stream_writer(config)
        all_follow_ups = state.get("follow_up_queries", [])
        if not all_follow_ups:
            return {"search_results": []}

        # Deduplicate: only search queries not already in search_results
        # (follow_up_queries accumulates across iterations via operator.add)
        existing_query_ids = {sr.get("sub_query_id", "") for sr in state.get("search_results", [])}
        follow_up_queries = [
            sq for sq in all_follow_ups if sq.get("id", "") not in existing_query_ids
        ]
        if not follow_up_queries:
            return {"search_results": []}
        import deep_research_swarm.backends.searxng  # noqa: F401

        if settings.exa_api_key:
            import deep_research_swarm.backends.exa  # noqa: F401
        if settings.tavily_api_key:
            import deep_research_swarm.backends.tavily  # noqa: F401

        if writer:
            writer(
                {
                    "kind": "search_progress",
                    "message": f"follow-up: {len(follow_up_queries)} queries",
                }
            )

        _snap = state.get("tunable_snapshot", {})
        num_results = int(_snap.get("results_per_query", 15))

        tasks = [
            search_sub_query(
                sq,
                backend_configs=backend_configs,
                cache=search_cache,
                num_results=num_results,
            )
            for sq in follow_up_queries
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for result in results_lists:
            if isinstance(result, list):
                all_results.extend(result)

        if writer:
            writer(
                {
                    "kind": "search_progress",
                    "message": "follow-up complete",
                    "count": len(all_results),
                }
            )

        return {"search_results": all_results}

    async def extract_followup_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """V9: Extract content from follow-up search results."""
        writer = _get_stream_writer(config)
        # Only extract results from follow-up queries
        follow_up_ids = {sq["id"] for sq in state.get("follow_up_queries", [])}
        all_results = state.get("search_results", [])
        followup_results = [r for r in all_results if r.get("sub_query_id") in follow_up_ids]

        if not followup_results:
            return {"extracted_contents": []}

        # Deduplicate by URL (against ALL previous extractions)
        already_extracted = {ec["url"] for ec in state.get("extracted_contents", [])}
        unique_results = []
        seen_urls: set[str] = set()
        for sr in followup_results:
            url = sr["url"]
            if url not in seen_urls and url not in already_extracted:
                seen_urls.add(url)
                unique_results.append(sr)

        _snap = state.get("tunable_snapshot", {})
        follow_up_budget = int(_snap.get("follow_up_budget", 5))
        content_trunc = int(_snap.get("content_truncation_chars", 50000))

        # Cap follow-up extractions to budget * 3
        capped = unique_results[: follow_up_budget * 3]

        if writer:
            writer({"kind": "extract_progress", "message": f"follow-up: {len(capped)} URLs"})

        sem = asyncio.Semaphore(settings.max_concurrent_requests)
        _grobid_url = settings.grobid_url

        async def extract_with_sem(sr):
            async with sem:
                return await extract_content(
                    sr, content_truncation_chars=content_trunc, grobid_url=_grobid_url
                )

        tasks = [extract_with_sem(sr) for sr in capped]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extracted = []
        for result in results:
            if isinstance(result, dict):
                extracted.append(result)

        if writer:
            writer(
                {
                    "kind": "extract_progress",
                    "message": "follow-up complete",
                    "count": len(extracted),
                }
            )

        return {"extracted_contents": extracted}

    async def score_merge_node(state: ResearchState) -> dict:
        """V9: Re-score all documents (original + follow-up) after follow-up extraction."""
        search_results = state.get("search_results", [])
        extracted_contents = state.get("extracted_contents", [])

        scored = build_scored_documents(
            search_results,
            extracted_contents,
            k=settings.rrf_k,
            authority_weight=settings.authority_weight,
        )
        diversity = compute_diversity(scored)

        # Also re-chunk with new content
        passages = chunk_all_documents(extracted_contents, scored)

        return {
            "scored_documents": scored,
            "diversity_metrics": diversity,
            "source_passages": passages,
        }

    async def citation_chain_node(state: ResearchState) -> dict:
        """Expand evidence via citation graph traversal (V7, PR-08)."""
        return await citation_chain(state, s2_backend)

    async def contradiction_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """Detect contradictions among scored documents."""
        scored_docs = state.get("scored_documents", [])
        if len(scored_docs) < 2:
            return {"contradictions": []}

        # Read adaptive tunable (V8) — fall back to 10
        _snap = state.get("tunable_snapshot", {})
        max_docs = int(_snap.get("contradiction_max_docs", 10))

        contradictions, usage = await detect_contradictions(
            scored_docs, sonnet_caller, max_docs=max_docs
        )
        result: dict = {"contradictions": contradictions}
        if usage:
            result["token_usage"] = [usage]

        # V9: Rich streaming — emit contradiction summary
        writer = _get_stream_writer(config)
        if writer and contradictions:
            writer(
                {
                    "kind": "contradiction_summary",
                    "count": len(contradictions),
                }
            )

        return result

    async def deliberate_panel_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """Run 4-judge deliberation panel (V10/Tensegrity)."""
        scored_docs = state.get("scored_documents", [])
        passages = state.get("source_passages", [])
        section_drafts = state.get("section_drafts", [])
        citation_map = state.get("citation_to_passage_map", {})
        research_question = state.get("research_question", "")
        sub_queries = state.get("sub_queries", [])
        diversity = state.get("diversity_metrics")

        # Run judges (authority + grounding + coverage are sync, contradiction is async)
        auth_result = authority_judge(scored_docs, passages)
        ground_result = grounding_judge(section_drafts, passages, citation_map)
        cover_result = coverage_judge(
            scored_docs,
            research_question,
            sub_queries,
            diversity,
        )

        _snap = state.get("tunable_snapshot", {})
        max_docs = int(_snap.get("contradiction_max_docs", 10))
        contra_result = await contradiction_judge(
            scored_docs,
            sonnet_caller,
            max_docs=max_docs,
        )

        # Merge judgments
        wave_num = state.get("wave_count", 0) + 1
        jc = merge_judgments(
            auth_result,
            ground_result,
            contra_result,
            cover_result,
            wave_number=wave_num,
        )

        result: dict = {
            "judgment_context": dict(jc),
            "panel_judgments": [
                {"judge": "authority", "summary": auth_result},
                {"judge": "grounding", "summary": ground_result},
                {"judge": "contradiction", "summary": contra_result},
                {"judge": "coverage", "summary": cover_result},
            ],
            "wave_count": wave_num,
        }

        # Merge token usage from contradiction judge
        if contra_result.get("token_usage"):
            result["token_usage"] = contra_result["token_usage"]

        # Stream panel completion
        writer = _get_stream_writer(config)
        if writer:
            try:
                writer(
                    {
                        "kind": "panel_complete",
                        "coverage": jc.get("overall_coverage", 0),
                        "risks": len(jc.get("structural_risks", [])),
                        "tensions": len(jc.get("active_tensions", [])),
                        "claims": len(jc.get("claim_verdicts", [])),
                    }
                )
            except Exception:
                pass

        return result

    async def compress_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """Build KnowledgeArtifact from judgments + passages (V10)."""
        jc = state.get("judgment_context", {})
        passages = state.get("source_passages", [])
        research_question = state.get("research_question", "")

        if not jc or not passages:
            return {}

        _snap = state.get("tunable_snapshot", {})
        max_clusters = int(_snap.get("max_clusters", 12))
        claims_per = int(_snap.get("claims_per_cluster", 8))

        artifact = build_knowledge_artifact(
            jc,
            passages,
            max_clusters=max_clusters,
            claims_per_cluster=claims_per,
            use_embeddings=True,
            embedding_model=settings.embedding_model,
        )
        artifact["question"] = research_question

        # Stream compression event
        writer = _get_stream_writer(config)
        if writer:
            try:
                writer(
                    {
                        "kind": "compression_complete",
                        "clusters": len(artifact.get("clusters", [])),
                        "claims": len(artifact.get("claim_verdicts", [])),
                        "ratio": artifact.get("compression_ratio", 0),
                    }
                )
            except Exception:
                pass

        return {"knowledge_artifact": dict(artifact)}

    async def synthesize_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        result = await synthesize(
            state,
            opus_caller,
            sonnet_caller=sonnet_caller,
            haiku_caller=haiku_caller,
        )

        # Stream reactor events if reactor ran
        reactor_trace = result.get("reactor_trace")
        if reactor_trace:
            writer = _get_stream_writer(config)
            if writer:
                try:
                    writer(
                        {
                            "kind": "reactor_complete",
                            "turns": reactor_trace.get("turns_executed", 0),
                            "constraints": reactor_trace.get("constraints_produced", 0),
                            "agents_used": reactor_trace.get("agents_used", []),
                            "final_entropy": reactor_trace.get("final_entropy", 0),
                        }
                    )
                except Exception:
                    pass

        return result

    async def critique_node(state: ResearchState) -> dict:
        return await critique(
            state, sonnet_caller, convergence_threshold=settings.convergence_threshold
        )

    async def compute_entropy_node(
        state: ResearchState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """Compute thermodynamic entropy from state observables (V10/SIA)."""
        prev_entropy = state.get("entropy_state") or None
        entropy = compute_entropy(state, prev_entropy)

        # Check entropy gate
        section_drafts = state.get("section_drafts", [])
        gate_ok, gate_reason = entropy_gate(entropy, section_drafts)

        # Check false convergence
        contradictions = state.get("contradictions", [])
        false_conv, false_reason = detect_false_convergence(
            entropy, section_drafts, contradictions, prev_entropy
        )

        # Check dominance
        entropy_history = state.get("entropy_history", [])
        dom, dom_reason = detect_dominance(entropy_history, section_drafts)

        # Singularity check (V10 Phase 4) — uses reactor_state if available
        # reactor_state has turn_log, constraints, active_frames, coalition_map
        # (NOT reactor_trace, which is metadata only)
        reactor_state_for_check = state.get("reactor_state", {})
        sing_safe, sing_reason, sing_details = True, "no_reactor", {}
        if reactor_state_for_check:
            sing_safe, sing_reason, sing_details = singularity_check(reactor_state_for_check)

        # If entropy blocks convergence, override critic's decision
        result: dict = {
            "entropy_state": dict(entropy),
            "entropy_history": [dict(entropy)],
        }

        if state.get("converged", False):
            # Max-iterations is an unconditional ceiling — never veto it
            conv_reason = state.get("convergence_reason", "")
            if "max_iterations_reached" in conv_reason or "budget" in conv_reason:
                pass  # absolute ceiling, no entropy override
            # Five-way convergence check (V10 Phase 4):
            # confidence AND entropy_gate AND NOT false_convergence
            # AND NOT dominance AND singularity_safe
            elif not gate_ok:
                result["converged"] = False
                result["convergence_reason"] = f"entropy_veto: {gate_reason}"
            elif false_conv:
                result["converged"] = False
                result["convergence_reason"] = f"false_convergence: {false_reason}"
            elif dom:
                result["converged"] = False
                result["convergence_reason"] = f"dominance_detected: {dom_reason}"
            elif not sing_safe:
                result["converged"] = False
                result["convergence_reason"] = f"singularity: {sing_reason}"

        # Stream entropy event if writer available
        writer = _get_stream_writer(config)
        if writer:
            try:
                writer(
                    {
                        "kind": "entropy_computed",
                        "e": entropy["e"],
                        "band": entropy["band"],
                        "components": {
                            "amb": entropy["e_amb"],
                            "conf": entropy["e_conf"],
                            "nov": entropy["e_nov"],
                            "trust": entropy["e_trust"],
                        },
                    }
                )
            except Exception:
                pass

            # Stream singularity check result
            if reactor_state_for_check:
                try:
                    writer(
                        {
                            "kind": "singularity_check",
                            "safe": sing_safe,
                            "reason": sing_reason,
                        }
                    )
                except Exception:
                    pass

            # Stream adversarial findings summary
            adv_findings = state.get("adversarial_findings", [])
            if adv_findings:
                try:
                    writer(
                        {
                            "kind": "adversarial_summary",
                            "findings_count": len(adv_findings),
                            "critical": sum(
                                1 for f in adv_findings if f.get("severity") == "critical"
                            ),
                            "significant": sum(
                                1 for f in adv_findings if f.get("severity") == "significant"
                            ),
                        }
                    )
                except Exception:
                    pass

        return result

    async def rollup_budget_node(state: ResearchState) -> dict:
        """Roll up token_usage list into totals for budget tracking."""
        usage_list = state.get("token_usage", [])
        total_tokens = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage_list)
        total_cost = sum(u.get("cost_usd", 0.0) for u in usage_list)
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
        "clarify": clarify_node,
        "plan": plan_node,
        "search": search_node,
        "adapt_extraction": adapt_extraction_node,
        "extract": extract_node,
        "chunk_passages": chunk_passages_node,
        "score": score_node,
        "gap_analysis": gap_analysis_node,
        "search_followup": search_followup_node,
        "extract_followup": extract_followup_node,
        "score_merge": score_merge_node,
        "adapt_synthesis": adapt_synthesis_node,
        "citation_chain": citation_chain_node,
        "contradiction": contradiction_node,
        "deliberate_panel": deliberate_panel_node,
        "compress": compress_node,
        "synthesize": synthesize_node,
        "critique": critique_node,
        "compute_entropy": compute_entropy_node,
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
    graph.add_edge("health_check", "clarify")
    graph.add_edge("clarify", "plan")

    if hitl_mode:
        graph.add_edge("plan", "plan_gate")
        graph.add_edge("plan_gate", "search")
    else:
        graph.add_edge("plan", "search")

    graph.add_edge("search", "adapt_extraction")
    graph.add_edge("adapt_extraction", "extract")
    graph.add_edge("extract", "chunk_passages")
    graph.add_edge("chunk_passages", "score")

    # V9: Reactive search loop (G2) — score -> gap_analysis -> [follow-up?] -> adapt_synthesis
    graph.add_edge("score", "gap_analysis")

    def should_follow_up(state: ResearchState) -> str:
        """Route after gap_analysis: follow-up search or continue to synthesis.

        Uses follow_up_round as the sole gate signal. gap_analyzer sets
        follow_up_round=1 when it produces new queries and returns []
        when follow_up_round > 0 (already ran this iteration). This avoids
        routing on the accumulated follow_up_queries list, which contains
        queries from all prior iterations.
        """
        follow_up_round = state.get("follow_up_round", 0)
        # gap_analyzer returns follow_up_round=1 when queries produced,
        # and [] when follow_up_round > 0 (duplicate guard).
        # The plan node resets follow_up_round=0 each iteration.
        # Route to follow-up only when gap_analyzer just produced queries.
        if follow_up_round == 1 and state.get("follow_up_queries", []):
            return "search_followup"
        return "adapt_synthesis"

    graph.add_conditional_edges(
        "gap_analysis",
        should_follow_up,
        {"search_followup": "search_followup", "adapt_synthesis": "adapt_synthesis"},
    )
    graph.add_edge("search_followup", "extract_followup")
    graph.add_edge("extract_followup", "score_merge")
    graph.add_edge("score_merge", "adapt_synthesis")

    graph.add_edge("adapt_synthesis", "citation_chain")
    graph.add_edge("citation_chain", "contradiction")
    graph.add_edge("contradiction", "deliberate_panel")
    graph.add_edge("deliberate_panel", "compress")
    graph.add_edge("compress", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_edge("critique", "compute_entropy")
    graph.add_edge("compute_entropy", "rollup_budget")

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

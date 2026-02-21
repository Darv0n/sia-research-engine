"""CLI entry point: python -m deep_research_swarm <question>"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from deep_research_swarm.config import get_settings
from deep_research_swarm.graph.builder import build_graph
from deep_research_swarm.streaming import StreamDisplay


def _generate_thread_id() -> str:
    """Generate a unique thread ID: research-YYYYMMDD-HHMMSS-XXXX."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(2).hex()
    return f"research-{ts}-{suffix}"


@asynccontextmanager
async def _make_checkpointer(settings, db_path: str):
    """Yield the appropriate checkpointer based on settings, or None.

    Handles sqlite, postgres, and none backends.
    """
    backend = settings.checkpoint_backend

    if backend == "none":
        yield None
        return

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:
            print(
                "ERROR: langgraph-checkpoint-postgres is not installed.\n"
                "Install with: pip install 'deep-research-swarm[postgres]'",
                file=sys.stderr,
            )
            sys.exit(1)
        async with AsyncPostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
        return

    # Default: sqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deep-research-swarm",
        description="Multi-agent deep research system",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="The research question to investigate",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum research iterations (default: from config)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Maximum token budget (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the report (default: stdout + output/<timestamp>.md)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=None,
        help="Search backends to use (default: all available)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable search result caching",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming output (use blocking ainvoke)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed progress during streaming",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="THREAD_ID",
        help="Resume a previous research run by thread ID",
    )
    parser.add_argument(
        "--list-threads",
        action="store_true",
        default=False,
        help="List recent research threads from checkpoint database",
    )
    # V5 flags
    parser.add_argument(
        "--dump-state",
        type=str,
        default=None,
        metavar="THREAD_ID",
        help="Export checkpoint state as JSON for a given thread ID",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="Disable memory store/retrieval for this run",
    )
    parser.add_argument(
        "--list-memories",
        action="store_true",
        default=False,
        help="Print stored memory records and exit",
    )
    parser.add_argument(
        "--export-mcp",
        action="store_true",
        default=False,
        help="Export memory records in MCP entity format and exit",
    )
    # V6 flags
    parser.add_argument(
        "--no-log",
        action="store_true",
        default=False,
        help="Disable run event logging",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "hitl"],
        default=None,
        help="Execution mode: 'auto' (default) or 'hitl' (human-in-the-loop gates)",
    )
    args = parser.parse_args()

    # Validate: need question, --resume, or a standalone flag
    standalone = args.list_threads or args.dump_state or args.list_memories or args.export_mcp
    if not args.question and not args.resume and not standalone:
        parser.error(
            "a question is required (or use --resume THREAD_ID / --list-threads "
            "/ --dump-state THREAD_ID / --list-memories / --export-mcp)"
        )

    return args


async def _execute(
    graph,
    input_state: dict | None,
    config: dict,
    *,
    no_stream: bool,
    verbose: bool,
) -> dict | None:
    """Execute the graph with streaming or blocking invocation.

    Returns the final state dict, or None if interrupted (HITL gate).
    """
    if no_stream:
        result = await graph.ainvoke(input_state, config=config)
        final_state = await graph.aget_state(config=config)
        if final_state and final_state.next:
            _print_interrupt_from_state(final_state, config)
            return None
        return result

    display = StreamDisplay(verbose=verbose)

    async for event in graph.astream(
        input_state,
        config=config,
        stream_mode=["updates", "custom"],
    ):
        if isinstance(event, tuple) and len(event) == 2:
            stream_mode, payload = event
            if stream_mode == "updates":
                display.handle_update(payload)
            elif stream_mode == "custom":
                display.handle_custom(payload)

    # Retrieve final state via checkpoint
    final_state = await graph.aget_state(config=config)
    if final_state and final_state.next:
        _print_interrupt_from_state(final_state, config)
        return None
    if final_state and hasattr(final_state, "values"):
        return final_state.values
    return {}


def _print_interrupt_from_state(state_snapshot, config: dict) -> None:
    """Print interrupt payload and resume instructions to stderr."""
    thread_id = config.get("configurable", {}).get("thread_id", "UNKNOWN")

    # Extract interrupt payload from tasks
    tasks = getattr(state_snapshot, "tasks", ())
    for task in tasks:
        interrupts = getattr(task, "interrupts", ())
        for intr in interrupts:
            payload = getattr(intr, "value", intr)
            gate = payload.get("gate", "unknown") if isinstance(payload, dict) else "unknown"
            message = payload.get("message", "") if isinstance(payload, dict) else str(payload)
            print(f"\n--- HITL Gate: {gate} ---", file=sys.stderr)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    if k not in ("gate", "message"):
                        if isinstance(v, list) and len(v) > 5:
                            print(f"  {k}: [{len(v)} items]", file=sys.stderr)
                        else:
                            print(f"  {k}: {v}", file=sys.stderr)
            if message:
                print(f"  {message}", file=sys.stderr)

    print(f"\nResume with: python -m deep_research_swarm --resume {thread_id}", file=sys.stderr)


def _output_report(result: dict, *, output_path_override: str | None, question: str) -> None:
    """Print report to stdout and save to file."""
    report = result.get("final_report", "")
    if not report:
        print("WARNING: No report generated.", file=sys.stderr)
        sys.exit(1)

    sys.stdout.buffer.write(report.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()

    # Save to file
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    if output_path_override:
        output_path = Path(output_path_override)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_q = "".join(c if c.isalnum() or c in "-_ " else "" for c in question[:50])
        safe_q = safe_q.strip().replace(" ", "-").lower()
        output_path = output_dir / f"{ts}-{safe_q}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}", file=sys.stderr)

    # Print summary
    total_tokens = result.get("total_tokens_used", 0)
    total_cost = result.get("total_cost_usd", 0.0)
    iterations = len(result.get("iteration_history", []))
    reason = result.get("convergence_reason", "")
    print(
        f"Completed: {iterations} iteration(s) | {total_tokens:,} tokens | "
        f"${total_cost:.4f} | Reason: {reason}",
        file=sys.stderr,
    )


async def _list_threads(settings, db_path: str) -> None:
    """List recent research threads from the checkpoint database."""
    if settings.checkpoint_backend == "none":
        print("Checkpointing is disabled.", file=sys.stderr)
        return

    if settings.checkpoint_backend == "sqlite" and not Path(db_path).exists():
        print("No checkpoint database found.", file=sys.stderr)
        return

    async with _make_checkpointer(settings, db_path) as checkpointer:
        if checkpointer is None:
            print("Checkpointing is disabled.", file=sys.stderr)
            return

        seen: dict[str, str] = {}  # thread_id -> status (first = latest checkpoint)
        async for cp in checkpointer.alist(None, limit=100):
            thread_id = cp.config["configurable"].get("thread_id", "unknown")
            if thread_id in seen:
                continue
            has_next = bool(cp.metadata.get("writes"))
            seen[thread_id] = "in-progress" if has_next else "completed"
        if not seen:
            print("No threads found.", file=sys.stderr)
            return
        for thread_id, status in seen.items():
            print(f"  {thread_id}  [{status}]")


async def _dump_state(settings, db_path: str, thread_id: str, output: str | None) -> None:
    """Export checkpoint state as JSON for a given thread ID."""
    if settings.checkpoint_backend == "none":
        print("ERROR: Checkpointing is disabled.", file=sys.stderr)
        sys.exit(1)

    if settings.checkpoint_backend == "sqlite" and not Path(db_path).exists():
        print(f"ERROR: Checkpoint database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    async with _make_checkpointer(settings, db_path) as checkpointer:
        if checkpointer is None:
            print("ERROR: No checkpointer available.", file=sys.stderr)
            sys.exit(1)

        graph = build_graph(settings, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        snapshot = await graph.aget_state(config=config)
        if not snapshot or not snapshot.values:
            print(f"ERROR: No checkpoint found for thread '{thread_id}'", file=sys.stderr)
            sys.exit(1)

        # Serialize state — use default=str for non-serializable types
        state_json = json.dumps(snapshot.values, indent=2, default=str, ensure_ascii=False)

        if output:
            Path(output).write_text(state_json, encoding="utf-8")
            print(f"State exported to: {output}", file=sys.stderr)
        else:
            print(state_json)


def _list_memories(memory_dir: str) -> None:
    """Print stored memory records."""
    from deep_research_swarm.memory.store import MemoryStore

    store = MemoryStore(memory_dir)
    records = store.list_all()

    if not records:
        print("No memory records found.", file=sys.stderr)
        return

    for rec in records:
        converged = "converged" if rec.get("converged") else "incomplete"
        findings_count = len(rec.get("key_findings", []))
        gaps_count = len(rec.get("gaps", []))
        print(
            f"  {rec.get('thread_id', '?')}  "
            f"[{converged}]  "
            f"{rec.get('iterations', 0)} iter, "
            f"{rec.get('sources_count', 0)} sources, "
            f"{findings_count} findings, "
            f"{gaps_count} gaps"
        )
        print(f"    Q: {rec.get('question', '?')}")
        print(f"    T: {rec.get('timestamp', '?')}")


def _export_mcp(memory_dir: str) -> None:
    """Export memory records in MCP entity format."""
    from deep_research_swarm.memory.mcp_export import export_to_mcp_format
    from deep_research_swarm.memory.store import MemoryStore

    store = MemoryStore(memory_dir)
    records = store.list_all()

    if not records:
        print("No memory records to export.", file=sys.stderr)
        return

    print(export_to_mcp_format(records))


def _format_memory_context(memories: list[dict]) -> str:
    """Format retrieved memories into a context string for the planner."""
    if not memories:
        return ""

    parts = []
    for mem in memories:
        lines = [f"- Prior research: {mem.get('question', '?')}"]
        findings = mem.get("key_findings", [])
        if findings:
            lines.append(f"  Findings: {', '.join(findings)}")
        gaps = mem.get("gaps", [])
        if gaps:
            lines.append(f"  Open gaps: {', '.join(gaps)}")
        parts.append("\n".join(lines))

    return "\n".join(parts)


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()

    # Validate
    errors = settings.validate()
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    db_path = str(project_root / settings.checkpoint_db)
    memory_dir = str(project_root / settings.memory_dir)
    run_log_dir = str(project_root / settings.run_log_dir)

    # Resolve mode (CLI flag overrides config)
    mode = args.mode or settings.mode

    # --- Standalone operations (no graph execution) ---

    if args.list_threads:
        await _list_threads(settings, db_path)
        return

    if args.dump_state:
        await _dump_state(settings, db_path, args.dump_state, args.output)
        return

    if args.list_memories:
        _list_memories(memory_dir)
        return

    if args.export_mcp:
        _export_mcp(memory_dir)
        return

    # --- Resume path ---

    if args.resume:
        if settings.checkpoint_backend == "none":
            print("ERROR: Cannot resume with checkpoint_backend='none'", file=sys.stderr)
            sys.exit(1)

        if settings.checkpoint_backend == "sqlite" and not Path(db_path).exists():
            print(f"ERROR: Checkpoint database not found: {db_path}", file=sys.stderr)
            sys.exit(1)

        async with _make_checkpointer(settings, db_path) as checkpointer:
            graph = build_graph(
                settings,
                enable_cache=not args.no_cache,
                checkpointer=checkpointer,
                mode=mode,
            )
            config = {"configurable": {"thread_id": args.resume}}

            snapshot = await graph.aget_state(config=config)
            if not snapshot or not snapshot.values:
                print(f"ERROR: No checkpoint found for thread '{args.resume}'", file=sys.stderr)
                sys.exit(1)

            if not snapshot.next:
                # Completed run — display stored report
                print(f"Thread '{args.resume}' already completed.", file=sys.stderr)
                question = snapshot.values.get("research_question", args.resume)
                _output_report(snapshot.values, output_path_override=args.output, question=question)
                return

            # Check if paused at a HITL gate — resume with Command
            paused_nodes = list(snapshot.next)
            is_gate = any("gate" in n for n in paused_nodes)

            if is_gate:
                from langgraph.types import Command

                print(
                    f"Resuming through HITL gate: {paused_nodes}",
                    file=sys.stderr,
                )
                resume_input = Command(resume=True)
            else:
                print(
                    f"Resuming thread '{args.resume}' from node(s): {paused_nodes}",
                    file=sys.stderr,
                )
                resume_input = None  # None input resumes from checkpoint

            result = await _execute(
                graph,
                resume_input,
                config,
                no_stream=args.no_stream,
                verbose=args.verbose,
            )
            if result is None:
                # Hit another HITL gate
                return
            question = result.get("research_question", args.resume)
            _output_report(result, output_path_override=args.output, question=question)
        return

    # --- New research run ---

    if not args.question:
        print("ERROR: A research question is required.", file=sys.stderr)
        sys.exit(1)

    backends = args.backends or settings.available_backends()
    max_iter = args.max_iterations or settings.max_iterations
    budget = args.token_budget or settings.token_budget

    # Event log (V6)
    event_log = None
    if not args.no_log:
        from deep_research_swarm.event_log.writer import EventLog

    # Memory retrieval (pre-computed context for planner)
    use_memory = not args.no_memory
    memory_context = ""
    store = None

    if use_memory:
        from deep_research_swarm.memory.store import MemoryStore

        store = MemoryStore(memory_dir)
        memories = store.search(args.question)
        memory_context = _format_memory_context(memories)
        if memory_context:
            print(f"Memory: loaded {len(memories)} prior research record(s)", file=sys.stderr)

    initial_state = {
        "research_question": args.question,
        "max_iterations": max_iter,
        "token_budget": budget,
        "search_backends": backends,
        "memory_context": memory_context,
        "perspectives": [],
        "sub_queries": [],
        "search_results": [],
        "extracted_contents": [],
        "scored_documents": [],
        "diversity_metrics": {},
        "section_drafts": [],
        "citations": [],
        "contradictions": [],
        "research_gaps": [],
        "current_iteration": 0,
        "converged": False,
        "convergence_reason": "",
        "token_usage": [],
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "iteration_history": [],
        "final_report": "",
    }

    use_checkpointer = settings.checkpoint_backend != "none"
    thread_id = _generate_thread_id()

    # Create event log after thread_id is known
    if not args.no_log:
        event_log = EventLog(run_log_dir, thread_id)

    if use_checkpointer:
        async with _make_checkpointer(settings, db_path) as checkpointer:
            graph = build_graph(
                settings,
                enable_cache=not args.no_cache,
                checkpointer=checkpointer,
                event_log=event_log,
                mode=mode,
            )
            config = {"configurable": {"thread_id": thread_id}}

            print(f"Thread: {thread_id}", file=sys.stderr)
            print(f"Researching: {args.question}", file=sys.stderr)
            print(f"Backends: {', '.join(backends)}", file=sys.stderr)
            print(f"Max iterations: {max_iter} | Token budget: {budget:,}", file=sys.stderr)
            print("---", file=sys.stderr)

            result = await _execute(
                graph,
                initial_state,
                config,
                no_stream=args.no_stream,
                verbose=args.verbose,
            )
    else:
        # No checkpointing
        graph = build_graph(
            settings,
            enable_cache=not args.no_cache,
            event_log=event_log,
            mode=mode,
        )
        config = {}

        print(f"Researching: {args.question}", file=sys.stderr)
        print(f"Backends: {', '.join(backends)}", file=sys.stderr)
        print(f"Max iterations: {max_iter} | Token budget: {budget:,}", file=sys.stderr)
        print("---", file=sys.stderr)

        result = await _execute(
            graph,
            initial_state,
            config,
            no_stream=args.no_stream,
            verbose=args.verbose,
        )

    if result is None:
        # Interrupted at HITL gate
        if event_log is not None:
            print(f"Event log: {event_log.path}", file=sys.stderr)
        return

    _output_report(result, output_path_override=args.output, question=args.question)

    # Print event log path
    if event_log is not None:
        print(f"Event log: {event_log.path}", file=sys.stderr)

    # Memory storage (after successful report output)
    if use_memory and store is not None:
        from deep_research_swarm.memory.extract import extract_memory_record

        record = extract_memory_record(result, thread_id)
        store.add_record(record)
        print(f"Memory: stored record for thread {thread_id}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

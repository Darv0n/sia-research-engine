"""CLI entry point: python -m deep_research_swarm <question>"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
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
    args = parser.parse_args()

    # Validate: need question, --resume, or --list-threads
    if not args.question and not args.resume and not args.list_threads:
        parser.error("a question is required (or use --resume THREAD_ID / --list-threads)")

    return args


async def _execute(
    graph,
    input_state: dict | None,
    config: dict,
    *,
    no_stream: bool,
    verbose: bool,
) -> dict:
    """Execute the graph with streaming or blocking invocation.

    Returns the final state dict.
    """
    if no_stream:
        return await graph.ainvoke(input_state, config=config)

    display = StreamDisplay(verbose=verbose)

    async for event in graph.astream(
        input_state,
        config=config,
        stream_mode=["updates", "custom"],
    ):
        if isinstance(event, tuple) and len(event) == 2:
            mode, payload = event
            if mode == "updates":
                display.handle_update(payload)
            elif mode == "custom":
                display.handle_custom(payload)

    # Retrieve final state via checkpoint
    final_state = await graph.aget_state(config=config)
    if final_state and hasattr(final_state, "values"):
        return final_state.values
    return {}


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


async def _list_threads(db_path: str) -> None:
    """List recent research threads from the checkpoint database."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    if not Path(db_path).exists():
        print("No checkpoint database found.", file=sys.stderr)
        return

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
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


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()

    # Validate
    errors = settings.validate()
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Resolve checkpoint database path
    project_root = Path(__file__).resolve().parent.parent
    db_path = str(project_root / settings.checkpoint_db)

    # --list-threads: show threads and exit
    if args.list_threads:
        await _list_threads(db_path)
        return

    use_checkpointer = settings.checkpoint_backend == "sqlite"

    # --resume: resume a previous thread
    if args.resume:
        if not use_checkpointer:
            print("ERROR: Cannot resume with checkpoint_backend='none'", file=sys.stderr)
            sys.exit(1)

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        if not Path(db_path).exists():
            print(f"ERROR: Checkpoint database not found: {db_path}", file=sys.stderr)
            sys.exit(1)

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            graph = build_graph(settings, enable_cache=not args.no_cache, checkpointer=checkpointer)
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

            # Incomplete run — resume from last checkpoint
            print(f"Resuming thread '{args.resume}' from node(s): {snapshot.next}", file=sys.stderr)
            result = await _execute(
                graph,
                None,  # None input resumes from checkpoint
                config,
                no_stream=args.no_stream,
                verbose=args.verbose,
            )
            question = result.get("research_question", args.resume)
            _output_report(result, output_path_override=args.output, question=question)
        return

    # New research run
    if not args.question:
        print("ERROR: A research question is required.", file=sys.stderr)
        sys.exit(1)

    backends = args.backends or settings.available_backends()
    max_iter = args.max_iterations or settings.max_iterations
    budget = args.token_budget or settings.token_budget

    initial_state = {
        "research_question": args.question,
        "max_iterations": max_iter,
        "token_budget": budget,
        "search_backends": backends,
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

    if use_checkpointer:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # Ensure checkpoint directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        thread_id = _generate_thread_id()

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            graph = build_graph(settings, enable_cache=not args.no_cache, checkpointer=checkpointer)
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
        # No checkpointing — run like V3
        graph = build_graph(settings, enable_cache=not args.no_cache)
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

    _output_report(result, output_path_override=args.output, question=args.question)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

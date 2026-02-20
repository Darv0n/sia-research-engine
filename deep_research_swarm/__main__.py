"""CLI entry point: python -m deep_research_swarm <question>"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from deep_research_swarm.config import get_settings
from deep_research_swarm.graph.builder import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deep-research-swarm",
        description="Multi-agent deep research system",
    )
    parser.add_argument(
        "question",
        type=str,
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
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()

    # Validate
    errors = settings.validate()
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Build graph
    graph = build_graph(settings)

    # Prepare initial state
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
        "section_drafts": [],
        "citations": [],
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

    print(f"Researching: {args.question}")
    print(f"Backends: {', '.join(backends)}")
    print(f"Max iterations: {max_iter} | Token budget: {budget:,}")
    print("---")

    # Run the graph
    result = await graph.ainvoke(initial_state)

    # Output report
    report = result.get("final_report", "")
    if not report:
        print("WARNING: No report generated.", file=sys.stderr)
        sys.exit(1)

    # Print to stdout
    print(report)

    # Save to file
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_q = "".join(c if c.isalnum() or c in "-_ " else "" for c in args.question[:50])
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


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

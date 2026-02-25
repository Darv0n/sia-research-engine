"""Multi-reactor swarm orchestrator.

Runs N parallel full-pipeline invocations with perturbed initialization.
Each reactor gets a different entropy-band seed, perspective ordering,
or query emphasis, producing genuinely different exploration paths.

Winner selection uses structural criteria:
  - Entropy stability (lower final entropy = more convergent)
  - Constraint density (more constraints = better-explored)
  - Grounding quality (average confidence score)
  - Cross-validation (findings confirmed by multiple reactors)

Usage:
  orchestrator = SwarmOrchestrator(settings, n_reactors=3)
  result = await orchestrator.run(question, initial_state, ...)
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
from typing import Any

from deep_research_swarm.config import Settings
from deep_research_swarm.contracts import SwarmMetadata
from deep_research_swarm.sia.reactor_coupling import (
    CouplingChannel,
    build_artifact_injection,
    build_entropy_broadcast,
    build_validation_shock,
)

# Perturbation strategies for reactor diversity
PERTURBATION_STRATEGIES = [
    "baseline",  # No perturbation — control reactor
    "entropy_high",  # Start with elevated entropy seed (turbulence band)
    "entropy_low",  # Start with low entropy seed (crystalline band)
    "perspective_shuffle",  # Shuffle perspective ordering
    "depth_focus",  # Fewer queries, deeper extraction per source
]


def _generate_reactor_id(index: int, question: str) -> str:
    """Generate a deterministic reactor ID."""
    h = hashlib.md5(f"{question}:{index}".encode()).hexdigest()[:6]
    return f"reactor-{index}-{h}"


def _perturb_state(
    base_state: dict[str, Any],
    strategy: str,
    reactor_index: int,
) -> dict[str, Any]:
    """Apply a perturbation strategy to create a variant initial state.

    Each strategy produces a meaningfully different exploration path
    while keeping the core question and budget intact.
    """
    state = copy.deepcopy(base_state)

    if strategy == "baseline":
        # No perturbation — serves as control
        pass

    elif strategy == "entropy_high":
        # Seed with elevated entropy to encourage broader exploration
        state["entropy_state"] = {
            "e": 0.55,
            "e_amb": 0.50,
            "e_conf": 0.40,
            "e_nov": 0.70,
            "e_trust": 0.55,
            "band": "turbulence",
            "turn": 0,
            "stagnation_count": 0,
        }

    elif strategy == "entropy_low":
        # Seed with low entropy to encourage faster convergence
        state["entropy_state"] = {
            "e": 0.15,
            "e_amb": 0.10,
            "e_conf": 0.10,
            "e_nov": 0.20,
            "e_trust": 0.15,
            "band": "crystalline",
            "turn": 0,
            "stagnation_count": 0,
        }

    elif strategy == "perspective_shuffle":
        # Reverse perspective ordering to change exploration priority
        perspectives = state.get("perspectives", [])
        if perspectives:
            state["perspectives"] = list(reversed(perspectives))

    elif strategy == "depth_focus":
        # Adjust tunables toward fewer, deeper results
        snapshot = state.get("tunable_snapshot", {})
        snapshot = dict(snapshot)
        snapshot["target_queries"] = max(6, snapshot.get("target_queries", 12) // 2)
        snapshot["results_per_query"] = min(25, snapshot.get("results_per_query", 15) + 5)
        snapshot["extraction_cap"] = min(80, snapshot.get("extraction_cap", 50) + 15)
        state["tunable_snapshot"] = snapshot

    return state


def _split_budget(total_budget: int, n_reactors: int) -> list[int]:
    """Split token budget across reactors.

    Each reactor gets an equal share, with remainder going to the first.
    """
    per_reactor = total_budget // n_reactors
    budgets = [per_reactor] * n_reactors
    budgets[0] += total_budget - (per_reactor * n_reactors)
    return budgets


def _score_reactor(result: dict[str, Any]) -> dict[str, float]:
    """Compute structural quality scores for a completed reactor.

    Returns a dict of named scores, each in [0, 1].
    """
    scores: dict[str, float] = {}

    # Entropy stability: lower final entropy = more convergent
    entropy_state = result.get("entropy_state", {})
    final_e = entropy_state.get("e", 0.5)
    scores["entropy_stability"] = max(0.0, 1.0 - final_e)

    # Constraint density: from reactor trace
    reactor_trace = result.get("reactor_trace", {})
    constraints = reactor_trace.get("constraints_produced", 0)
    # Normalize: 10+ constraints = 1.0
    scores["constraint_density"] = min(1.0, constraints / 10.0)

    # Grounding quality: average confidence across sections
    section_drafts = result.get("section_drafts", [])
    if section_drafts:
        avg_conf = sum(s.get("confidence_score", 0.5) for s in section_drafts) / len(section_drafts)
        scores["grounding_quality"] = avg_conf
    else:
        scores["grounding_quality"] = 0.0

    # Section count: more sections = more comprehensive (up to a point)
    scores["comprehensiveness"] = min(1.0, len(section_drafts) / 8.0)

    # Convergence: did it converge naturally?
    scores["convergence"] = 1.0 if result.get("converged", False) else 0.3

    return scores


def _compute_cross_validation(
    results: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Cross-validate reactors against each other.

    A reactor's cross-validation score is higher when its key claims
    and constraints are confirmed by other reactors' findings.
    """
    if len(results) < 2:
        return {rid: 0.5 for rid in results}

    cv_scores: dict[str, float] = {}

    for rid, result in results.items():
        # Get this reactor's section headings as a proxy for coverage
        headings = {s.get("heading", "").lower() for s in result.get("section_drafts", [])}

        # Check overlap with other reactors
        overlap_count = 0
        total_others = 0

        for other_rid, other_result in results.items():
            if other_rid == rid:
                continue
            total_others += 1

            other_headings = {
                s.get("heading", "").lower() for s in other_result.get("section_drafts", [])
            }

            # Heading overlap (structural agreement)
            if headings and other_headings:
                intersection = headings & other_headings
                overlap_ratio = len(intersection) / max(len(headings), 1)
                overlap_count += overlap_ratio

        cv_scores[rid] = overlap_count / max(total_others, 1)

    return cv_scores


def select_winner(
    results: dict[str, dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> tuple[str, str, dict[str, float]]:
    """Select the best reactor from completed results.

    Returns (winner_id, reason, selection_scores).

    Default weights:
      entropy_stability: 0.20
      constraint_density: 0.15
      grounding_quality: 0.30
      comprehensiveness: 0.10
      convergence: 0.10
      cross_validation: 0.15
    """
    if not results:
        return "", "no_results", {}

    if len(results) == 1:
        rid = next(iter(results))
        return rid, "single_reactor", {rid: 1.0}

    default_weights = {
        "entropy_stability": 0.20,
        "constraint_density": 0.15,
        "grounding_quality": 0.30,
        "comprehensiveness": 0.10,
        "convergence": 0.10,
        "cross_validation": 0.15,
    }
    w = weights or default_weights

    # Score each reactor
    reactor_scores: dict[str, dict[str, float]] = {}
    for rid, result in results.items():
        reactor_scores[rid] = _score_reactor(result)

    # Add cross-validation scores
    cv_scores = _compute_cross_validation(results)
    for rid in reactor_scores:
        reactor_scores[rid]["cross_validation"] = cv_scores.get(rid, 0.5)

    # Weighted aggregate
    final_scores: dict[str, float] = {}
    for rid, scores in reactor_scores.items():
        total = sum(scores.get(dim, 0.0) * w.get(dim, 0.0) for dim in w)
        final_scores[rid] = round(total, 4)

    # Winner
    winner_id = max(final_scores, key=final_scores.get)
    runner_up = sorted(final_scores, key=final_scores.get, reverse=True)
    margin = 0.0
    if len(runner_up) >= 2:
        margin = final_scores[runner_up[0]] - final_scores[runner_up[1]]

    reason = f"highest_weighted_score ({final_scores[winner_id]:.3f}, margin={margin:.3f})"

    return winner_id, reason, final_scores


class SwarmOrchestrator:
    """Orchestrates N parallel reactor runs with perturbed initialization.

    Usage::

        orchestrator = SwarmOrchestrator(settings, n_reactors=3)
        result = await orchestrator.run(question, initial_state)
    """

    def __init__(
        self,
        settings: Settings,
        n_reactors: int = 3,
    ):
        if n_reactors < 2:
            raise ValueError("Swarm requires at least 2 reactors for cross-validation")
        if n_reactors > settings.swarm_max_reactors:
            n_reactors = settings.swarm_max_reactors

        self.settings = settings
        self.n_reactors = n_reactors
        self.coupling = CouplingChannel()

    def _build_reactor_configs(
        self,
        base_state: dict[str, Any],
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Build (reactor_id, strategy, perturbed_state) for each reactor."""
        question = base_state.get("research_question", "")
        total_budget = base_state.get("token_budget", 200000)
        budgets = _split_budget(total_budget, self.n_reactors)

        configs = []
        for i in range(self.n_reactors):
            reactor_id = _generate_reactor_id(i, question)
            strategy = PERTURBATION_STRATEGIES[i % len(PERTURBATION_STRATEGIES)]
            perturbed = _perturb_state(base_state, strategy, i)
            perturbed["token_budget"] = budgets[i]
            configs.append((reactor_id, strategy, perturbed))

        return configs

    async def _run_single_reactor(
        self,
        reactor_id: str,
        strategy: str,
        initial_state: dict[str, Any],
        *,
        enable_cache: bool = True,
        mode: str = "auto",
        reactor_configs: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any] | None:
        """Run a single reactor (full pipeline). Returns result or None on failure."""
        from deep_research_swarm.graph.builder import build_graph

        try:
            graph = build_graph(
                self.settings,
                enable_cache=enable_cache,
                mode=mode,
            )

            result = await graph.ainvoke(initial_state, config={})

            # Post-run: generate coupling messages for other reactors.
            # NOTE: In parallel execution, messages are only useful for
            # post-hoc analysis or future staggered execution. receive()
            # is not called during parallel runs — this is scaffolding
            # for V11 staggered swarm mode.
            configs = reactor_configs or []
            other_ids = [rid for rid, _, _ in configs if rid != reactor_id]

            # Artifact injection
            artifact = result.get("knowledge_artifact", {})
            if artifact:
                msgs = build_artifact_injection(reactor_id, artifact, other_ids)
                for m in msgs:
                    self.coupling.send(m)

            # Entropy broadcast
            entropy = result.get("entropy_state", {})
            if entropy:
                msgs = build_entropy_broadcast(reactor_id, entropy, other_ids)
                for m in msgs:
                    self.coupling.send(m)

            # Validation shock
            findings = result.get("adversarial_findings", [])
            if findings:
                msgs = build_validation_shock(reactor_id, findings, other_ids)
                for m in msgs:
                    self.coupling.send(m)

            return result

        except Exception:
            return None

    async def run(
        self,
        base_state: dict[str, Any],
        *,
        enable_cache: bool = True,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """Run N reactors in parallel and select the winner.

        Returns the winning reactor's result dict, augmented with
        SwarmMetadata under the "swarm_metadata" key.
        """
        reactor_configs = self._build_reactor_configs(base_state)

        # Launch all reactors concurrently
        tasks = {
            reactor_id: asyncio.create_task(
                self._run_single_reactor(
                    reactor_id,
                    strategy,
                    state,
                    enable_cache=enable_cache,
                    mode=mode,
                    reactor_configs=reactor_configs,
                )
            )
            for reactor_id, strategy, state in reactor_configs
        }

        # Gather results (allow failures)
        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Pair results with reactor IDs
        completed: dict[str, dict[str, Any]] = {}
        failed: list[str] = []

        for (reactor_id, strategy, _state), result in zip(reactor_configs, raw_results):
            if isinstance(result, BaseException) or result is None:
                failed.append(reactor_id)
            else:
                completed[reactor_id] = result

        if not completed:
            # All failed — return a minimal error result
            return {
                "converged": False,
                "convergence_reason": "swarm_all_reactors_failed",
                "swarm_metadata": SwarmMetadata(
                    n_reactors=self.n_reactors,
                    reactor_configs=[{"id": rid, "strategy": s} for rid, s, _ in reactor_configs],
                    reactor_entropies=[],
                    reactor_tokens=[],
                    reactor_costs=[],
                    winner_id="",
                    selection_reason="all_failed",
                    selection_scores={},
                    cross_validation_scores={},
                    total_tokens_all=0,
                    total_cost_all=0.0,
                    failed_reactors=failed,
                ),
            }

        # Select winner
        winner_id, reason, selection_scores = select_winner(completed)
        cv_scores = _compute_cross_validation(completed)

        # Aggregate metadata
        reactor_entropies = [r.get("entropy_state", {}).get("e", 0.0) for r in completed.values()]
        reactor_tokens = [r.get("total_tokens_used", 0) for r in completed.values()]
        reactor_costs = [r.get("total_cost_usd", 0.0) for r in completed.values()]

        metadata = SwarmMetadata(
            n_reactors=self.n_reactors,
            reactor_configs=[{"id": rid, "strategy": s} for rid, s, _ in reactor_configs],
            reactor_entropies=reactor_entropies,
            reactor_tokens=reactor_tokens,
            reactor_costs=reactor_costs,
            winner_id=winner_id,
            selection_reason=reason,
            selection_scores=selection_scores,
            cross_validation_scores=cv_scores,
            total_tokens_all=sum(reactor_tokens),
            total_cost_all=sum(reactor_costs),
            failed_reactors=failed,
        )

        # Return winner's result augmented with swarm metadata
        winner_result = dict(completed[winner_id])
        winner_result["swarm_metadata"] = dict(metadata)

        return winner_result

"""Entropy-band steering — adjusts tunables based on thermodynamic state.

Deterministic, zero LLM calls. Reads current entropy band and adjusts
tunable_snapshot values to steer the pipeline:

  runaway:     reduce query volume, zero follow-up budget (contain)
  turbulence:  reduce target queries (moderate)
  convergence: deeper drafts, more sections (build)
  crystalline: no-op (harvest)
"""

from __future__ import annotations

from typing import Any

from deep_research_swarm.contracts import EntropyBand, EntropyState


def steer_tunables(
    entropy: EntropyState,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return adjusted tunable snapshot based on entropy band.

    Does NOT modify the input snapshot. Returns a new dict with adjustments.
    Only adjusts values that are present in the snapshot (backward compatible).
    """
    band = entropy["band"]
    adjusted = dict(snapshot)

    if band == EntropyBand.RUNAWAY.value:
        _apply_runaway(adjusted)
    elif band == EntropyBand.TURBULENCE.value:
        _apply_turbulence(adjusted)
    elif band == EntropyBand.CONVERGENCE.value:
        _apply_convergence(adjusted)
    # crystalline: no adjustments

    return adjusted


def _apply_runaway(snap: dict[str, Any]) -> None:
    """Runaway band: contain — reduce exploration, increase compression."""
    if "perspectives_count" in snap:
        snap["perspectives_count"] = max(2, snap["perspectives_count"] - 2)
    if "target_queries" in snap:
        snap["target_queries"] = max(3, snap["target_queries"] // 2)
    if "follow_up_budget" in snap:
        snap["follow_up_budget"] = 0


def _apply_turbulence(snap: dict[str, Any]) -> None:
    """Turbulence band: moderate — reduce query volume, keep follow-ups."""
    if "target_queries" in snap:
        snap["target_queries"] = max(5, snap["target_queries"] - 3)


def _apply_convergence(snap: dict[str, Any]) -> None:
    """Convergence band: build — deeper drafts, more sections."""
    if "max_sections" in snap:
        snap["max_sections"] = min(
            snap.get("max_sections", 8) + 1,
            15,  # ceiling from registry
        )

"""TunableRegistry — bounded parameter store for adaptive control.

Every tunable has a floor, ceiling, and default. The overseer adjusts values
within these bounds. Values outside bounds are clamped silently.

Design invariant I5: The overseer CANNOT disable safety mechanisms.
Floors prevent thresholds from going dangerously low.
"""

from __future__ import annotations

from typing import Any

from deep_research_swarm.contracts import Tunable

# --- Default tunable definitions ---

_DEFAULTS: list[Tunable] = [
    # Extraction
    Tunable(name="extraction_cap", default=50, floor=15, ceiling=150, category="extraction"),
    Tunable(
        name="content_truncation_chars",
        default=50000,
        floor=20000,
        ceiling=200000,
        category="extraction",
    ),
    # Scoring
    Tunable(name="contradiction_max_docs", default=10, floor=5, ceiling=30, category="scoring"),
    Tunable(
        name="budget_exhaustion_pct",
        default=0.9,
        floor=0.7,
        ceiling=0.95,
        category="scoring",
    ),
    # Grounding
    Tunable(name="jaccard_threshold", default=0.3, floor=0.15, ceiling=0.5, category="grounding"),
    Tunable(
        name="grounding_pass_threshold",
        default=0.8,
        floor=0.6,
        ceiling=0.95,
        category="grounding",
    ),
    Tunable(name="max_refinement_attempts", default=2, floor=1, ceiling=5, category="grounding"),
    Tunable(
        name="max_passages_per_section",
        default=10,
        floor=4,
        ceiling=20,
        category="grounding",
    ),
    # Search
    Tunable(name="citation_chain_budget", default=50, floor=20, ceiling=150, category="search"),
    Tunable(name="citation_chain_max_hops", default=2, floor=1, ceiling=4, category="search"),
    Tunable(name="citation_chain_top_seeds", default=5, floor=3, ceiling=15, category="search"),
    Tunable(name="results_per_query", default=15, floor=5, ceiling=30, category="search"),
    # Planning (V9)
    Tunable(name="perspectives_count", default=5, floor=3, ceiling=8, category="planning"),
    Tunable(name="target_queries", default=12, floor=6, ceiling=25, category="planning"),
    Tunable(name="follow_up_budget", default=5, floor=0, ceiling=10, category="planning"),
    # Synthesis
    Tunable(name="min_sections", default=3, floor=2, ceiling=5, category="synthesis"),
    Tunable(name="max_sections", default=8, floor=4, ceiling=15, category="synthesis"),
    Tunable(name="max_docs_for_outline", default=20, floor=10, ceiling=50, category="synthesis"),
    # Deliberation (V10)
    Tunable(name="max_waves", default=5, floor=2, ceiling=8, category="deliberation"),
    Tunable(name="wave_batch_size", default=3, floor=1, ceiling=5, category="deliberation"),
    Tunable(name="wave_extract_cap", default=15, floor=5, ceiling=30, category="deliberation"),
    Tunable(
        name="coverage_threshold", default=0.75, floor=0.5, ceiling=0.95, category="deliberation"
    ),
    Tunable(name="max_clusters", default=12, floor=3, ceiling=20, category="deliberation"),
    Tunable(name="claims_per_cluster", default=8, floor=3, ceiling=15, category="deliberation"),
    # SIA Reactor (V10 Phase 3)
    Tunable(name="sia_reactor_turns", default=6, floor=3, ceiling=10, category="synthesis"),
    Tunable(
        name="sia_reactor_budget", default=20000, floor=8000, ceiling=40000, category="synthesis"
    ),
]


def _clamp(value: int | float, floor: int | float, ceiling: int | float) -> int | float:
    """Clamp value within [floor, ceiling], preserving type."""
    clamped = max(floor, min(ceiling, value))
    # Preserve int type when floor/ceiling are ints
    if isinstance(floor, int) and isinstance(ceiling, int) and isinstance(value, int):
        return int(clamped)
    return clamped


class TunableRegistry:
    """Bounded parameter store for pipeline tunables.

    Thread-safe for read; mutations happen only in overseer nodes
    which run sequentially in the graph.
    """

    def __init__(self) -> None:
        self._tunables: dict[str, Tunable] = {}
        self._values: dict[str, int | float] = {}
        # Load built-in defaults
        for t in _DEFAULTS:
            self.register(t)

    def register(self, tunable: Tunable) -> None:
        """Register a tunable definition and set its value to default."""
        self._tunables[tunable["name"]] = tunable
        self._values[tunable["name"]] = tunable["default"]

    def get(self, name: str) -> int | float:
        """Get current value of a tunable. Raises KeyError if unknown."""
        return self._values[name]

    def get_default(self, name: str) -> int | float:
        """Get the default value of a tunable. Raises KeyError if unknown."""
        return self._tunables[name]["default"]

    def get_definition(self, name: str) -> Tunable:
        """Get the full tunable definition. Raises KeyError if unknown."""
        return self._tunables[name]

    def set(self, name: str, value: int | float) -> int | float:
        """Set a tunable's value, clamping to [floor, ceiling].

        Returns the actual (possibly clamped) value stored.
        Raises KeyError if the tunable name is not registered.
        """
        t = self._tunables[name]
        clamped = _clamp(value, t["floor"], t["ceiling"])
        self._values[name] = clamped
        return clamped

    def set_scaled(self, name: str, multiplier: float) -> int | float:
        """Scale a tunable's default by multiplier, clamp, and store.

        Returns the actual (possibly clamped) value stored.
        """
        t = self._tunables[name]
        raw = t["default"] * multiplier
        # Preserve int type for int defaults
        if isinstance(t["default"], int):
            raw = round(raw)
        return self.set(name, raw)

    def names(self) -> list[str]:
        """Return all registered tunable names."""
        return list(self._tunables.keys())

    def categories(self) -> dict[str, list[str]]:
        """Return tunables grouped by category."""
        cats: dict[str, list[str]] = {}
        for t in self._tunables.values():
            cats.setdefault(t["category"], []).append(t["name"])
        return cats

    def snapshot(self) -> dict[str, Any]:
        """Serialize current values to a dict (for state persistence)."""
        return dict(self._values)

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> TunableRegistry:
        """Restore a registry from a snapshot dict.

        Unknown keys in the snapshot are ignored (forward compat).
        Known tunables not in the snapshot keep their defaults.
        """
        registry = cls()
        for name, value in snap.items():
            if name in registry._tunables:
                registry.set(name, value)
        return registry

    def diff_from_defaults(self) -> dict[str, dict[str, Any]]:
        """Return only tunables whose current value differs from default.

        Returns {name: {"default": ..., "current": ..., "category": ...}}.
        """
        diffs: dict[str, dict[str, Any]] = {}
        for name, t in self._tunables.items():
            current = self._values[name]
            if current != t["default"]:
                diffs[name] = {
                    "default": t["default"],
                    "current": current,
                    "category": t["category"],
                }
        return diffs

    def __len__(self) -> int:
        return len(self._tunables)

    def __contains__(self, name: str) -> bool:
        return name in self._tunables

"""Token budget tracking across all agents."""

from __future__ import annotations

from deep_research_swarm.contracts import TokenUsage


class BudgetTracker:
    """Tracks cumulative token usage and enforces budget limits."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self._usage: list[TokenUsage] = []

    def record(self, usage: TokenUsage) -> None:
        self._usage.append(usage)

    @property
    def total_tokens(self) -> int:
        return sum(u["input_tokens"] + u["output_tokens"] for u in self._usage)

    @property
    def total_cost(self) -> float:
        return sum(u["cost_usd"] for u in self._usage)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.total_tokens)

    def has_budget(self, estimated_tokens: int = 0) -> bool:
        """Check if there's enough budget for the next operation."""
        return self.total_tokens + estimated_tokens <= self.budget

    def usage_by_agent(self) -> dict[str, dict[str, int | float]]:
        """Summarize usage per agent."""
        by_agent: dict[str, dict] = {}
        for u in self._usage:
            agent = u["agent"]
            if agent not in by_agent:
                by_agent[agent] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            by_agent[agent]["input_tokens"] += u["input_tokens"]
            by_agent[agent]["output_tokens"] += u["output_tokens"]
            by_agent[agent]["cost_usd"] += u["cost_usd"]
        return by_agent

    def summary(self) -> str:
        """Human-readable budget summary."""
        total = self.total_tokens
        pct = (total / self.budget * 100) if self.budget > 0 else 0
        return (
            f"Tokens: {total:,}/{self.budget:,} ({pct:.1f}%) | "
            f"Cost: ${self.total_cost:.4f} | "
            f"Remaining: {self.remaining:,}"
        )

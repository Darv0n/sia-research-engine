"""Tests for budget tracker."""

from deep_research_swarm.budget.tracker import BudgetTracker
from deep_research_swarm.contracts import TokenUsage


def _make_usage(agent: str, input_t: int, output_t: int, cost: float) -> TokenUsage:
    return TokenUsage(
        agent=agent,
        model="claude-opus-4-6",
        input_tokens=input_t,
        output_tokens=output_t,
        cost_usd=cost,
        timestamp="2026-02-20T00:00:00Z",
    )


class TestBudgetTracker:
    def test_initial_state(self):
        tracker = BudgetTracker(budget=100000)
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.remaining == 100000

    def test_record_usage(self):
        tracker = BudgetTracker(budget=100000)
        tracker.record(_make_usage("planner", 1000, 500, 0.03))
        assert tracker.total_tokens == 1500
        assert tracker.total_cost == 0.03
        assert tracker.remaining == 98500

    def test_multiple_records(self):
        tracker = BudgetTracker(budget=100000)
        tracker.record(_make_usage("planner", 1000, 500, 0.03))
        tracker.record(_make_usage("synthesizer", 2000, 1000, 0.06))
        assert tracker.total_tokens == 4500
        assert tracker.remaining == 95500

    def test_has_budget(self):
        tracker = BudgetTracker(budget=5000)
        tracker.record(_make_usage("planner", 3000, 1500, 0.10))
        assert tracker.has_budget(estimated_tokens=0) is True
        assert tracker.has_budget(estimated_tokens=500) is True
        assert tracker.has_budget(estimated_tokens=600) is False

    def test_budget_exceeded(self):
        tracker = BudgetTracker(budget=1000)
        tracker.record(_make_usage("planner", 800, 300, 0.05))
        assert tracker.remaining == 0  # Clamped to 0
        assert tracker.has_budget() is False

    def test_usage_by_agent(self):
        tracker = BudgetTracker(budget=100000)
        tracker.record(_make_usage("planner", 1000, 500, 0.03))
        tracker.record(_make_usage("planner", 500, 200, 0.01))
        tracker.record(_make_usage("synthesizer", 2000, 1000, 0.06))

        by_agent = tracker.usage_by_agent()
        assert "planner" in by_agent
        assert by_agent["planner"]["input_tokens"] == 1500
        assert "synthesizer" in by_agent

    def test_summary_format(self):
        tracker = BudgetTracker(budget=100000)
        tracker.record(_make_usage("planner", 1000, 500, 0.03))
        summary = tracker.summary()
        assert "1,500" in summary
        assert "100,000" in summary
        assert "$" in summary

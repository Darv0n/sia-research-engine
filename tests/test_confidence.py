"""Tests for confidence classification and replan logic."""

from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft
from deep_research_swarm.scoring.confidence import (
    aggregate_grader_scores,
    classify_confidence,
    should_replan,
    summarize_confidence,
)


def _make_section(score: float, level: Confidence, section_id: str = "sec-001") -> SectionDraft:
    return SectionDraft(
        id=section_id,
        heading="Test Section",
        content="Test content",
        citation_ids=[],
        confidence_score=score,
        confidence_level=level,
        grader_scores=GraderScores(relevance=score, hallucination=score, quality=score),
    )


class TestClassifyConfidence:
    def test_high(self):
        assert classify_confidence(0.85) == Confidence.HIGH
        assert classify_confidence(0.80) == Confidence.HIGH
        assert classify_confidence(1.0) == Confidence.HIGH

    def test_medium(self):
        assert classify_confidence(0.7) == Confidence.MEDIUM
        assert classify_confidence(0.6) == Confidence.MEDIUM
        assert classify_confidence(0.79) == Confidence.MEDIUM

    def test_low(self):
        assert classify_confidence(0.5) == Confidence.LOW
        assert classify_confidence(0.0) == Confidence.LOW
        assert classify_confidence(0.59) == Confidence.LOW


class TestAggregateGraderScores:
    def test_equal_weights(self):
        scores = GraderScores(relevance=0.9, hallucination=0.8, quality=0.7)
        assert abs(aggregate_grader_scores(scores) - 0.8) < 0.001

    def test_perfect_scores(self):
        scores = GraderScores(relevance=1.0, hallucination=1.0, quality=1.0)
        assert aggregate_grader_scores(scores) == 1.0

    def test_zero_scores(self):
        scores = GraderScores(relevance=0.0, hallucination=0.0, quality=0.0)
        assert aggregate_grader_scores(scores) == 0.0


class TestShouldReplan:
    def test_any_low_triggers_replan(self):
        sections = [
            _make_section(0.85, Confidence.HIGH, "sec-001"),
            _make_section(0.5, Confidence.LOW, "sec-002"),
        ]
        replan, reason = should_replan(sections)
        assert replan is True
        assert "LOW" in reason

    def test_all_high_no_replan(self):
        sections = [
            _make_section(0.9, Confidence.HIGH, "sec-001"),
            _make_section(0.85, Confidence.HIGH, "sec-002"),
        ]
        replan, reason = should_replan(sections)
        assert replan is False

    def test_over_30_percent_medium_triggers_replan(self):
        sections = [
            _make_section(0.7, Confidence.MEDIUM, "sec-001"),
            _make_section(0.65, Confidence.MEDIUM, "sec-002"),
            _make_section(0.85, Confidence.HIGH, "sec-003"),
        ]
        # 2/3 = 67% MEDIUM -> should replan
        replan, reason = should_replan(sections)
        assert replan is True
        assert "MEDIUM" in reason

    def test_empty_sections_replan(self):
        replan, reason = should_replan([])
        assert replan is True
        assert reason == "no_sections"

    def test_diminishing_returns(self):
        sections = [
            _make_section(0.85, Confidence.HIGH, "sec-001"),
            _make_section(0.82, Confidence.HIGH, "sec-002"),
        ]
        # prev_avg very close to current -> diminishing returns
        replan, reason = should_replan(sections, prev_avg=0.83)
        assert replan is False
        assert "diminishing_returns" in reason


class TestSummarizeConfidence:
    def test_counts(self, sample_section_drafts):
        counts = summarize_confidence(sample_section_drafts)
        assert counts["HIGH"] == 1
        assert counts["MEDIUM"] == 1
        assert counts["LOW"] == 0

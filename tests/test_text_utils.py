"""Tests for utils.text — Jaccard scoring and duplicate detection."""

from deep_research_swarm.utils.text import is_duplicate, jaccard_score


class TestJaccardScore:
    def test_identical_strings(self):
        assert jaccard_score("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert jaccard_score("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        # {hello, world} & {hello, there} = {hello}
        # union = {hello, world, there} -> 1/3
        score = jaccard_score("hello world", "hello there")
        assert abs(score - 1 / 3) < 0.001

    def test_case_insensitive(self):
        assert jaccard_score("Hello World", "hello world") == 1.0

    def test_empty_string_a(self):
        assert jaccard_score("", "hello") == 0.0

    def test_empty_string_b(self):
        assert jaccard_score("hello", "") == 0.0

    def test_both_empty(self):
        assert jaccard_score("", "") == 0.0


class TestIsDuplicate:
    """Mirrors test_query_dedup.py but uses the extracted utils.text function."""

    def test_identical_queries_are_duplicate(self):
        assert is_duplicate("quantum entanglement", ["quantum entanglement"])

    def test_case_insensitive(self):
        assert is_duplicate("Quantum Entanglement", ["quantum entanglement"])

    def test_high_overlap_not_duplicate(self):
        assert not is_duplicate(
            "quantum entanglement experiments",
            ["quantum entanglement research"],
        )

    def test_substring_containment_detected(self):
        assert is_duplicate("quantum", ["quantum entanglement"])

    def test_no_overlap(self):
        assert not is_duplicate(
            "history of Rome",
            ["quantum entanglement", "machine learning basics"],
        )

    def test_empty_existing_list(self):
        assert not is_duplicate("quantum entanglement", [])

    def test_empty_query_is_duplicate(self):
        assert is_duplicate("", ["anything"])

    def test_threshold_boundary(self):
        assert is_duplicate("a b c d e f g", ["a b c d e f g h i j"])

    def test_custom_threshold(self):
        # With threshold 0.3: {hello, world} & {hello, there} = 1/3 ≈ 0.33 >= 0.3
        assert is_duplicate("hello world", ["hello there"], threshold=0.3)
        # With threshold 0.5: 1/3 < 0.5, and no substring match
        assert not is_duplicate("hello world", ["hello there"], threshold=0.5)

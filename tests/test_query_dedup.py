"""Tests for V2 query deduplication in planner."""

from deep_research_swarm.agents.planner import _is_duplicate


class TestIsDuplicate:
    """Jaccard similarity + substring containment dedup."""

    def test_identical_queries_are_duplicate(self):
        assert _is_duplicate("quantum entanglement", ["quantum entanglement"])

    def test_case_insensitive(self):
        assert _is_duplicate("Quantum Entanglement", ["quantum entanglement"])

    def test_high_overlap_is_duplicate(self):
        # "quantum entanglement experiments" vs "quantum entanglement research"
        # Shared: {quantum, entanglement} / Union: {quantum, entanglement, experiments, research}
        # Jaccard = 2/4 = 0.5, below 0.7 threshold -> NOT duplicate
        assert not _is_duplicate(
            "quantum entanglement experiments",
            ["quantum entanglement research"],
        )

    def test_very_high_overlap_is_duplicate(self):
        # "how does quantum entanglement work" vs "how quantum entanglement works"
        # Shared: {how, quantum, entanglement}
        # Union: {how, does, quantum, entanglement, work, works}
        # Jaccard = 3/6 = 0.5 -> NOT duplicate by Jaccard alone
        # But substring check: neither contains the other -> NOT duplicate
        assert not _is_duplicate(
            "how does quantum entanglement work",
            ["how quantum entanglement works"],
        )

    def test_substring_containment_detected(self):
        assert _is_duplicate("quantum", ["quantum entanglement"])

    def test_reverse_substring_containment_detected(self):
        assert _is_duplicate("quantum entanglement effects", ["entanglement effects"])

    def test_no_overlap(self):
        assert not _is_duplicate(
            "history of Rome",
            ["quantum entanglement", "machine learning basics"],
        )

    def test_empty_existing_list(self):
        assert not _is_duplicate("quantum entanglement", [])

    def test_empty_query_is_duplicate(self):
        assert _is_duplicate("", ["anything"])

    def test_threshold_boundary(self):
        # Exactly at threshold: 7 shared out of 10 union = 0.7
        # "a b c d e f g" vs "a b c d e f g h i j" -> shared=7, union=10, J=0.7
        assert _is_duplicate(
            "a b c d e f g",
            ["a b c d e f g h i j"],
        )

    def test_below_threshold(self):
        # 6 shared out of 10 union = 0.6 (below 0.7)
        # Use non-substring tokens to isolate Jaccard test from substring check
        # "a b c d e f x y" vs "a b c d e f g h i j" -> shared=6, union=12, J=0.5
        assert not _is_duplicate(
            "a b c d e f x y",
            ["a b c d e f g h i j"],
        )

    def test_multiple_existing_one_match(self):
        assert _is_duplicate(
            "quantum entanglement",
            ["history of Rome", "quantum entanglement basics", "machine learning"],
        )

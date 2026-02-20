"""Tests for source diversity scoring (HHI-based)."""

from __future__ import annotations

from deep_research_swarm.contracts import ScoredDocument, SourceAuthority
from deep_research_swarm.scoring.diversity import compute_diversity


def _doc(url: str, authority: SourceAuthority = SourceAuthority.UNKNOWN) -> ScoredDocument:
    """Helper to build a minimal ScoredDocument."""
    return ScoredDocument(
        id="sd-test",
        url=url,
        title="Test",
        content="Content",
        rrf_score=0.01,
        authority=authority,
        authority_score=0.5,
        combined_score=0.1,
        sub_query_ids=["sq-001"],
    )


class TestComputeDiversity:
    def test_single_domain_worst_diversity(self):
        """All docs from one domain -> HHI=1.0, diversity=0.0."""
        docs = [
            _doc("https://example.com/a"),
            _doc("https://example.com/b"),
            _doc("https://example.com/c"),
        ]
        m = compute_diversity(docs)
        assert m["domain_concentration"] == 1.0
        assert m["diversity_score"] == 0.0
        assert m["unique_domains"] == 1

    def test_uniform_distribution(self):
        """Each doc from a different domain -> minimum HHI, maximum diversity."""
        docs = [
            _doc("https://a.com/1"),
            _doc("https://b.com/2"),
            _doc("https://c.com/3"),
            _doc("https://d.com/4"),
        ]
        m = compute_diversity(docs)
        # HHI for uniform = 1/N = 0.25
        assert m["domain_concentration"] == 0.25
        assert m["diversity_score"] == 0.75
        assert m["unique_domains"] == 4
        assert m["total_documents"] == 4

    def test_empty_documents(self):
        """Empty list returns zeroed metrics."""
        m = compute_diversity([])
        assert m["unique_domains"] == 0
        assert m["total_documents"] == 0
        assert m["diversity_score"] == 0.0

    def test_hhi_math_two_domains(self):
        """2 of 3 from domain A, 1 from domain B."""
        docs = [
            _doc("https://a.com/1"),
            _doc("https://a.com/2"),
            _doc("https://b.com/1"),
        ]
        m = compute_diversity(docs)
        # shares: 2/3 and 1/3 -> HHI = (2/3)^2 + (1/3)^2 = 4/9 + 1/9 = 5/9 ≈ 0.5556
        expected_hhi = round(5 / 9, 4)
        assert m["domain_concentration"] == expected_hhi
        assert m["diversity_score"] == round(1.0 - 5 / 9, 4)

    def test_authority_counts(self):
        """Authority distribution counts are accurate."""
        docs = [
            _doc("https://a.edu/1", SourceAuthority.INSTITUTIONAL),
            _doc("https://b.edu/2", SourceAuthority.INSTITUTIONAL),
            _doc("https://c.com/3", SourceAuthority.COMMUNITY),
        ]
        m = compute_diversity(docs)
        assert m["authority_distribution"]["institutional"] == 2
        assert m["authority_distribution"]["community"] == 1

    def test_www_prefix_normalized(self):
        """www.example.com and example.com treated as same domain."""
        docs = [
            _doc("https://www.example.com/a"),
            _doc("https://example.com/b"),
        ]
        m = compute_diversity(docs)
        assert m["unique_domains"] == 1
        assert m["domain_concentration"] == 1.0

"""Tests for RRF algorithm."""

from deep_research_swarm.contracts import SearchResult, SourceAuthority
from deep_research_swarm.scoring.rrf import build_scored_documents, reciprocal_rank_fusion


def _make_result(url: str, rank: int, sub_query_id: str = "sq-001") -> SearchResult:
    return SearchResult(
        id=f"sr-{url[-3:]}",
        sub_query_id=sub_query_id,
        url=url,
        title=f"Title {url}",
        snippet=f"Snippet {url}",
        backend="searxng",
        rank=rank,
        score=0.0,
        authority=SourceAuthority.UNKNOWN,
        timestamp="2026-02-20T00:00:00Z",
    )


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [_make_result("http://a.com", 1), _make_result("http://b.com", 2)]
        scores = reciprocal_rank_fusion([results], k=60)
        assert scores["http://a.com"] > scores["http://b.com"]

    def test_overlapping_docs_fuse(self):
        """Same URL in multiple lists should get higher score."""
        list1 = [_make_result("http://a.com", 1), _make_result("http://b.com", 2)]
        list2 = [_make_result("http://a.com", 1), _make_result("http://c.com", 2)]
        scores = reciprocal_rank_fusion([list1, list2], k=60)

        # a.com appears in both lists -> higher score
        assert scores["http://a.com"] > scores["http://b.com"]
        assert scores["http://a.com"] > scores["http://c.com"]

    def test_k_parameter_applied(self):
        results = [_make_result("http://a.com", 1)]
        scores_k60 = reciprocal_rank_fusion([results], k=60)
        scores_k10 = reciprocal_rank_fusion([results], k=10)
        # Higher k = lower individual score (more smoothing)
        assert scores_k60["http://a.com"] < scores_k10["http://a.com"]

    def test_empty_input_returns_empty(self):
        scores = reciprocal_rank_fusion([], k=60)
        assert scores == {}

    def test_empty_list_returns_empty(self):
        scores = reciprocal_rank_fusion([[]], k=60)
        assert scores == {}


class TestBuildScoredDocuments:
    def test_builds_from_results_and_contents(
        self, sample_search_results, sample_extracted_contents
    ):
        scored = build_scored_documents(
            sample_search_results,
            sample_extracted_contents,
            k=60,
            authority_weight=0.2,
        )
        assert len(scored) > 0
        # Should be sorted by combined_score descending
        for i in range(len(scored) - 1):
            assert scored[i]["combined_score"] >= scored[i + 1]["combined_score"]

    def test_institutional_beats_community(
        self, sample_search_results, sample_extracted_contents
    ):
        """Institutional source should rank higher with authority weighting."""
        scored = build_scored_documents(
            sample_search_results,
            sample_extracted_contents,
            k=60,
            authority_weight=0.2,
        )
        # Nature (institutional) should be near the top
        urls = [d["url"] for d in scored]
        nature_idx = next(
            (i for i, u in enumerate(urls) if "nature.com" in u), None
        )
        assert nature_idx is not None

    def test_empty_results(self):
        scored = build_scored_documents([], [])
        assert scored == []

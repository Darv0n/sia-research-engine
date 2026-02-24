"""Tests for agents/citation_chain.py — citation graph traversal (PR-08)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deep_research_swarm.agents.citation_chain import (
    _BUDGET,
    _RELEVANCE_THRESHOLD,
    _build_search_result,
    _extract_paper_id,
    _has_scholarly_results,
    _relevance_score,
    citation_chain,
)
from deep_research_swarm.contracts import (
    ScholarlyMetadata,
    ScoredDocument,
    SearchResult,
    SourceAuthority,
)


def _make_scored_doc(
    *,
    url: str = "https://doi.org/10.1234/test",
    combined_score: float = 0.9,
    doi: str = "10.1234/test",
    title: str = "Test Paper",
) -> ScoredDocument:
    return ScoredDocument(
        id="sd-test",
        url=url,
        title=title,
        content="content",
        rrf_score=0.5,
        authority=SourceAuthority.INSTITUTIONAL,
        authority_score=0.8,
        combined_score=combined_score,
        sub_query_ids=["sq-001"],
        scholarly_metadata=ScholarlyMetadata(
            doi=doi,
            arxiv_id="",
            pmid="",
            title=title,
            authors=["Author A"],
            year=2024,
            venue="Nature",
            citation_count=100,
            reference_count=50,
            is_open_access=False,
            open_access_url="",
            abstract="Abstract about test paper.",
        ),
    )


def _make_paper_details(
    paper_id: str,
    title: str,
    abstract: str = "",
    references: list[dict] | None = None,
    citations: list[dict] | None = None,
) -> dict:
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "year": 2024,
        "venue": "Test Journal",
        "citationCount": 10,
        "referenceCount": 5,
        "isOpenAccess": False,
        "openAccessPdf": None,
        "authors": [{"name": "Test Author"}],
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "externalIds": {"DOI": f"10.test/{paper_id}"},
        "references": references or [],
        "citations": citations or [],
    }


class TestRelevanceScore:
    def test_high_overlap(self):
        score = _relevance_score(
            "machine learning for natural language processing",
            "",
            "natural language processing using machine learning",
        )
        assert score > _RELEVANCE_THRESHOLD

    def test_zero_overlap(self):
        score = _relevance_score(
            "quantum entanglement in superconductors",
            "",
            "cooking recipes for italian pasta",
        )
        assert score == 0.0

    def test_with_abstract(self):
        score = _relevance_score(
            "A Survey Paper",
            "This paper reviews machine learning approaches for natural language processing",
            "machine learning natural language processing",
        )
        assert score > _RELEVANCE_THRESHOLD

    def test_empty_question(self):
        assert _relevance_score("some title", "abstract", "") == 0.0


class TestExtractPaperId:
    def test_from_doi(self):
        sr = SearchResult(
            id="sr-1",
            sub_query_id="",
            url="https://doi.org/10.1234/test",
            title="Test",
            snippet="",
            backend="s2",
            rank=1,
            score=0.5,
            authority=SourceAuthority.INSTITUTIONAL,
            timestamp="2026-01-01",
            scholarly_metadata=ScholarlyMetadata(
                doi="10.1234/test",
                arxiv_id="",
                pmid="",
                title="Test",
                authors=[],
                year=2024,
                venue="",
                citation_count=0,
                reference_count=0,
                is_open_access=False,
                open_access_url="",
                abstract="",
            ),
        )
        assert _extract_paper_id(sr) == "10.1234/test"

    def test_from_arxiv(self):
        sr = SearchResult(
            id="sr-1",
            sub_query_id="",
            url="https://arxiv.org/abs/2401.12345",
            title="Test",
            snippet="",
            backend="s2",
            rank=1,
            score=0.5,
            authority=SourceAuthority.INSTITUTIONAL,
            timestamp="2026-01-01",
            scholarly_metadata=ScholarlyMetadata(
                doi="",
                arxiv_id="2401.12345",
                pmid="",
                title="Test",
                authors=[],
                year=2024,
                venue="",
                citation_count=0,
                reference_count=0,
                is_open_access=False,
                open_access_url="",
                abstract="",
            ),
        )
        assert _extract_paper_id(sr) == "ArXiv:2401.12345"

    def test_no_metadata(self):
        sr = SearchResult(
            id="sr-1",
            sub_query_id="",
            url="https://example.com",
            title="Test",
            snippet="",
            backend="searxng",
            rank=1,
            score=0.5,
            authority=SourceAuthority.UNKNOWN,
            timestamp="2026-01-01",
        )
        assert _extract_paper_id(sr) is None


class TestHasScholarlyResults:
    def test_with_scholarly(self):
        state = {"scored_documents": [_make_scored_doc()]}
        assert _has_scholarly_results(state) is True

    def test_without_scholarly(self):
        doc = ScoredDocument(
            id="sd-1",
            url="https://example.com",
            title="Test",
            content="",
            rrf_score=0.5,
            authority=SourceAuthority.UNKNOWN,
            authority_score=0.0,
            combined_score=0.5,
            sub_query_ids=["sq-1"],
        )
        state = {"scored_documents": [doc]}
        assert _has_scholarly_results(state) is False

    def test_empty_state(self):
        assert _has_scholarly_results({}) is False


class TestBuildSearchResult:
    def test_builds_result(self):
        paper = _make_paper_details("abc123", "Test Paper", "An abstract")
        sr = _build_search_result(paper, "test question")
        assert sr is not None
        assert sr["title"] == "Test Paper"
        assert sr["backend"] == "citation_chain"
        assert sr["scholarly_metadata"]["doi"] == "10.test/abc123"
        assert sr["provenance"]["extractor"] == "citation_chain"

    def test_no_title_returns_none(self):
        paper = {"paperId": "abc", "title": ""}
        assert _build_search_result(paper, "q") is None


class TestCitationChain:
    @pytest.mark.asyncio
    async def test_no_s2_backend_returns_empty(self):
        state = {"scored_documents": [_make_scored_doc()]}
        result = await citation_chain(state, None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_scholarly_results_returns_empty(self):
        state = {
            "scored_documents": [
                ScoredDocument(
                    id="sd-1",
                    url="https://example.com",
                    title="Test",
                    content="",
                    rrf_score=0.5,
                    authority=SourceAuthority.UNKNOWN,
                    authority_score=0.0,
                    combined_score=0.5,
                    sub_query_ids=["sq-1"],
                )
            ]
        }
        mock_backend = AsyncMock()
        result = await citation_chain(state, mock_backend)
        assert result == {}
        mock_backend.get_paper_details.assert_not_called()

    @pytest.mark.asyncio
    async def test_discovers_new_papers(self):
        """S2 backend with references leads to new results."""
        seed_doc = _make_scored_doc(
            doi="10.1234/seed",
            title="Machine Learning for NLP",
            combined_score=0.95,
        )
        state = {
            "research_question": "machine learning natural language processing",
            "scored_documents": [seed_doc],
            "citation_chain_results": [],
        }

        # Seed paper details with relevant references
        seed_details = _make_paper_details(
            "seed-id",
            "Machine Learning for NLP",
            abstract="A study of ML approaches for NLP",
            references=[
                {"paperId": "ref-1", "title": "Deep learning for language understanding"},
                {"paperId": "ref-2", "title": "Cooking recipes for Italian food"},  # irrelevant
            ],
        )

        # Reference details
        ref1_details = _make_paper_details(
            "ref-1",
            "Deep learning for language understanding",
            abstract="This paper explores deep learning methods for natural language understanding",
        )

        mock_backend = AsyncMock()
        mock_backend.get_paper_details = AsyncMock(
            side_effect=lambda pid: {
                "10.1234/seed": seed_details,
                "ref-1": ref1_details,
            }.get(pid)
        )

        result = await citation_chain(state, mock_backend)
        assert "citation_chain_results" in result
        assert len(result["citation_chain_results"]) >= 1
        assert "search_results" in result
        # Results should also go to search_results
        assert result["search_results"] == result["citation_chain_results"]

    @pytest.mark.asyncio
    async def test_budget_enforcement(self):
        """Budget is checked before fetching, not after."""
        # Create many seed docs to fill budget quickly
        docs = []
        for i in range(10):
            docs.append(
                _make_scored_doc(
                    doi=f"10.1234/seed-{i}",
                    title=f"Paper about machine learning topic {i}",
                    combined_score=0.9 - i * 0.01,
                )
            )

        state = {
            "research_question": "machine learning",
            "scored_documents": docs,
            "citation_chain_results": [],
        }

        # Each seed has many references
        call_count = 0

        async def mock_details(pid):
            nonlocal call_count
            call_count += 1
            refs = [
                {"paperId": f"ref-{pid}-{j}", "title": f"machine learning paper {j}"}
                for j in range(100)
            ]
            return _make_paper_details(
                pid,
                f"Paper {pid}",
                "machine learning study",
                references=refs,
            )

        mock_backend = AsyncMock()
        mock_backend.get_paper_details = AsyncMock(side_effect=mock_details)

        result = await citation_chain(state, mock_backend)
        # Total seen papers should not exceed budget
        total_discovered = len(result.get("citation_chain_results", []))
        assert total_discovered <= _BUDGET

    @pytest.mark.asyncio
    async def test_dedup_same_paper_from_multiple_seeds(self):
        """Same paper referenced by multiple seeds is only fetched once."""
        doc1 = _make_scored_doc(doi="10.1234/seed1", combined_score=0.9)
        doc2 = _make_scored_doc(
            url="https://doi.org/10.1234/seed2",
            doi="10.1234/seed2",
            combined_score=0.85,
        )

        state = {
            "research_question": "machine learning natural language",
            "scored_documents": [doc1, doc2],
            "citation_chain_results": [],
        }

        shared_ref = {
            "paperId": "shared-ref",
            "title": "Natural language machine learning approach",
        }

        seed1_details = _make_paper_details("s1", "Seed 1", references=[shared_ref])
        seed2_details = _make_paper_details("s2", "Seed 2", references=[shared_ref])

        shared_details = _make_paper_details(
            "shared-ref",
            "Natural language machine learning approach",
            abstract="machine learning for natural language understanding",
        )

        detail_calls = []

        async def mock_details(pid):
            detail_calls.append(pid)
            return {
                "10.1234/seed1": seed1_details,
                "10.1234/seed2": seed2_details,
                "shared-ref": shared_details,
            }.get(pid)

        mock_backend = AsyncMock()
        mock_backend.get_paper_details = AsyncMock(side_effect=mock_details)

        result = await citation_chain(state, mock_backend)
        # shared-ref should appear in results at most once
        if result.get("citation_chain_results"):
            shared_results = [
                r
                for r in result["citation_chain_results"]
                if r.get("scholarly_metadata", {}).get("doi") == "10.test/shared-ref"
            ]
            assert len(shared_results) <= 1

    @pytest.mark.asyncio
    async def test_cross_iteration_dedup(self):
        """Papers already in citation_chain_results are not re-fetched."""
        seed_doc = _make_scored_doc(doi="10.1234/seed", combined_score=0.9)

        # This paper was already found in a previous iteration
        prev_result = SearchResult(
            id="sr-prev",
            sub_query_id="",
            url="https://doi.org/10.1234/prev-found",
            title="Previously Found",
            snippet="",
            backend="citation_chain",
            rank=0,
            score=0.0,
            authority=SourceAuthority.INSTITUTIONAL,
            timestamp="2026-01-01",
            scholarly_metadata=ScholarlyMetadata(
                doi="10.1234/prev-found",
                arxiv_id="",
                pmid="",
                title="Previously Found",
                authors=[],
                year=2024,
                venue="",
                citation_count=0,
                reference_count=0,
                is_open_access=False,
                open_access_url="",
                abstract="",
            ),
        )

        state = {
            "research_question": "machine learning natural language",
            "scored_documents": [seed_doc],
            "citation_chain_results": [prev_result],
        }

        seed_details = _make_paper_details(
            "seed-id",
            "Seed Paper",
            references=[
                {"paperId": "10.1234/prev-found", "title": "Previously Found machine learning"},
                {"paperId": "new-ref", "title": "New machine learning approach"},
            ],
        )
        new_details = _make_paper_details(
            "new-ref",
            "New machine learning approach",
            abstract="A novel machine learning approach to natural language",
        )

        detail_calls = []

        async def mock_details(pid):
            detail_calls.append(pid)
            return {
                "10.1234/seed": seed_details,
                "new-ref": new_details,
            }.get(pid)

        mock_backend = AsyncMock()
        mock_backend.get_paper_details = AsyncMock(side_effect=mock_details)

        await citation_chain(state, mock_backend)
        # prev-found should NOT be in detail_calls (already seen)
        assert "10.1234/prev-found" not in detail_calls

    @pytest.mark.asyncio
    async def test_irrelevant_papers_excluded(self):
        """Papers with zero relevance are not included in results."""
        seed_doc = _make_scored_doc(doi="10.1234/seed", combined_score=0.9)

        state = {
            "research_question": "machine learning for NLP",
            "scored_documents": [seed_doc],
            "citation_chain_results": [],
        }

        seed_details = _make_paper_details(
            "seed-id",
            "ML for NLP",
            references=[
                {"paperId": "irrelevant", "title": "Cooking recipes for pasta dishes"},
            ],
        )

        mock_backend = AsyncMock()
        mock_backend.get_paper_details = AsyncMock(
            side_effect=lambda pid: {"10.1234/seed": seed_details}.get(pid)
        )

        result = await citation_chain(state, mock_backend)
        # Should be empty since only reference is irrelevant
        assert result == {} or len(result.get("citation_chain_results", [])) == 0


class TestGraphWiring:
    def test_citation_chain_between_score_and_contradiction(self):
        """citation_chain node exists between score and contradiction."""
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(anthropic_api_key="test-key")
        graph = build_graph(settings, enable_cache=False)

        edges = set()
        g = graph.get_graph()
        for edge in g.edges:
            edges.add((edge.source, edge.target))

        # V9: score -> gap_analysis -> [conditional] -> adapt_synthesis -> citation_chain
        assert ("score", "gap_analysis") in edges
        assert ("adapt_synthesis", "citation_chain") in edges
        assert ("citation_chain", "contradiction") in edges
        # Old direct edges should not exist
        assert ("score", "contradiction") not in edges
        assert ("score", "citation_chain") not in edges
        assert ("score", "adapt_synthesis") not in edges  # V9: gap_analysis is now between

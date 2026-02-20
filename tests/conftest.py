"""Test fixtures and mocks."""

from __future__ import annotations

import pytest

from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    ExtractedContent,
    GraderScores,
    ScoredDocument,
    SearchResult,
    SectionDraft,
    SourceAuthority,
    SubQuery,
    TokenUsage,
)


@pytest.fixture
def sample_sub_query() -> SubQuery:
    return SubQuery(
        id="sq-001",
        question="What is quantum entanglement?",
        perspective="physics fundamentals",
        priority=1,
        parent_query_id=None,
        search_backends=["searxng"],
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            id="sr-001",
            sub_query_id="sq-001",
            url="https://en.wikipedia.org/wiki/Quantum_entanglement",
            title="Quantum entanglement - Wikipedia",
            snippet="Quantum entanglement is a phenomenon...",
            backend="searxng",
            rank=1,
            score=0.016,
            authority=SourceAuthority.COMMUNITY,
            timestamp="2026-02-20T00:00:00Z",
        ),
        SearchResult(
            id="sr-002",
            sub_query_id="sq-001",
            url="https://www.nature.com/articles/quantum-entanglement",
            title="Quantum Entanglement Explained - Nature",
            snippet="A comprehensive overview of quantum entanglement...",
            backend="searxng",
            rank=2,
            score=0.015,
            authority=SourceAuthority.INSTITUTIONAL,
            timestamp="2026-02-20T00:00:00Z",
        ),
        SearchResult(
            id="sr-003",
            sub_query_id="sq-002",
            url="https://arxiv.org/abs/2401.12345",
            title="Recent Advances in Entanglement",
            snippet="We review recent experimental advances...",
            backend="searxng",
            rank=1,
            score=0.016,
            authority=SourceAuthority.INSTITUTIONAL,
            timestamp="2026-02-20T00:00:00Z",
        ),
    ]


@pytest.fixture
def sample_extracted_contents() -> list[ExtractedContent]:
    return [
        ExtractedContent(
            id="ec-001",
            search_result_id="sr-001",
            url="https://en.wikipedia.org/wiki/Quantum_entanglement",
            title="Quantum entanglement - Wikipedia",
            content="Quantum entanglement is a phenomenon in quantum mechanics...",
            content_length=500,
            extractor_used="trafilatura",
            extraction_success=True,
            error=None,
        ),
        ExtractedContent(
            id="ec-002",
            search_result_id="sr-002",
            url="https://www.nature.com/articles/quantum-entanglement",
            title="Quantum Entanglement Explained - Nature",
            content="Entanglement is one of the most striking features...",
            content_length=800,
            extractor_used="crawl4ai",
            extraction_success=True,
            error=None,
        ),
    ]


@pytest.fixture
def sample_scored_documents() -> list[ScoredDocument]:
    return [
        ScoredDocument(
            id="sd-001",
            url="https://www.nature.com/articles/quantum-entanglement",
            title="Quantum Entanglement Explained - Nature",
            content="Entanglement is one of the most striking features...",
            rrf_score=0.032,
            authority=SourceAuthority.INSTITUTIONAL,
            authority_score=0.95,
            combined_score=0.216,
            sub_query_ids=["sq-001"],
        ),
        ScoredDocument(
            id="sd-002",
            url="https://en.wikipedia.org/wiki/Quantum_entanglement",
            title="Quantum entanglement - Wikipedia",
            content="Quantum entanglement is a phenomenon...",
            rrf_score=0.016,
            authority=SourceAuthority.COMMUNITY,
            authority_score=0.50,
            combined_score=0.113,
            sub_query_ids=["sq-001"],
        ),
    ]


@pytest.fixture
def sample_section_drafts() -> list[SectionDraft]:
    return [
        SectionDraft(
            id="sec-001",
            heading="Introduction to Quantum Entanglement",
            content="Quantum entanglement is a fundamental phenomenon [1]...",
            citation_ids=["[1]"],
            confidence_score=0.85,
            confidence_level=Confidence.HIGH,
            grader_scores=GraderScores(relevance=0.9, hallucination=0.85, quality=0.8),
        ),
        SectionDraft(
            id="sec-002",
            heading="Experimental Evidence",
            content="Bell test experiments have confirmed [2]...",
            citation_ids=["[2]"],
            confidence_score=0.75,
            confidence_level=Confidence.MEDIUM,
            grader_scores=GraderScores(relevance=0.8, hallucination=0.7, quality=0.75),
        ),
    ]


@pytest.fixture
def sample_citations() -> list[Citation]:
    return [
        Citation(
            id="[1]",
            url="https://www.nature.com/articles/quantum-entanglement",
            title="Quantum Entanglement Explained",
            authority=SourceAuthority.INSTITUTIONAL,
            accessed="2026-02-20T00:00:00Z",
            used_in_sections=["Introduction to Quantum Entanglement"],
        ),
        Citation(
            id="[2]",
            url="https://arxiv.org/abs/2401.12345",
            title="Recent Advances in Entanglement",
            authority=SourceAuthority.INSTITUTIONAL,
            accessed="2026-02-20T00:00:00Z",
            used_in_sections=["Experimental Evidence"],
        ),
    ]


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    return TokenUsage(
        agent="planner",
        model="claude-opus-4-6",
        input_tokens=1500,
        output_tokens=500,
        cost_usd=0.06,
        timestamp="2026-02-20T00:00:00Z",
    )

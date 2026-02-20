"""Single source of truth for all types, enums, and protocols."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, TypedDict, runtime_checkable

# --- Enums ---


class Confidence(str, Enum):
    HIGH = "HIGH"  # >= 0.8
    MEDIUM = "MEDIUM"  # 0.6 - 0.8
    LOW = "LOW"  # < 0.6


class SourceAuthority(str, Enum):
    INSTITUTIONAL = "institutional"  # .edu, .gov, journals
    PROFESSIONAL = "professional"  # news, tech blogs
    COMMUNITY = "community"  # forums, wikis
    PROMOTIONAL = "promotional"  # marketing
    UNKNOWN = "unknown"


# --- Data Types ---


class SubQuery(TypedDict):
    id: str  # "sq-001"
    question: str
    perspective: str
    priority: int  # 1 (highest) to 5
    parent_query_id: str | None
    search_backends: list[str]


class SearchResult(TypedDict):
    id: str
    sub_query_id: str
    url: str
    title: str
    snippet: str
    backend: str  # "searxng" | "exa" | "tavily"
    rank: int
    score: float  # 0-1
    authority: SourceAuthority
    timestamp: str  # ISO 8601


class ExtractedContent(TypedDict):
    id: str
    search_result_id: str
    url: str
    title: str
    content: str  # Cleaned markdown
    content_length: int
    extractor_used: str  # "crawl4ai" | "trafilatura" | "pymupdf4llm"
    extraction_success: bool
    error: str | None


class ScoredDocument(TypedDict):
    id: str
    url: str
    title: str
    content: str
    rrf_score: float
    authority: SourceAuthority
    authority_score: float  # 0-1
    combined_score: float  # rrf * (1-w) + authority * w
    sub_query_ids: list[str]


class GraderScores(TypedDict):
    relevance: float  # 0-1
    hallucination: float  # 0-1 (1.0 = no hallucination)
    quality: float  # 0-1


class SectionDraft(TypedDict):
    id: str
    heading: str
    content: str  # Markdown with [N] citations
    citation_ids: list[str]
    confidence_score: float
    confidence_level: Confidence
    grader_scores: GraderScores


class Citation(TypedDict):
    id: str  # "[1]", "[2]"
    url: str
    title: str
    authority: SourceAuthority
    accessed: str
    used_in_sections: list[str]


class ResearchGap(TypedDict):
    description: str
    attempted_queries: list[str]
    reason: str  # "no_sources" | "contradictory" | "low_confidence"


class TokenUsage(TypedDict):
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str


class Contradiction(TypedDict):
    claim_a: str
    source_a_url: str
    claim_b: str
    source_b_url: str
    topic: str
    severity: str  # "direct" | "nuanced" | "contextual"


class DiversityMetrics(TypedDict):
    unique_domains: int
    total_documents: int
    domain_concentration: float  # HHI: 0 (perfectly diverse) to 1 (single domain)
    authority_distribution: dict[str, int]  # authority level -> count
    diversity_score: float  # 0 (worst) to 1 (best), normalized from HHI


class SectionConfidenceSnapshot(TypedDict):
    heading: str
    confidence_score: float
    confidence_level: str  # Confidence enum value as string


class IterationRecord(TypedDict):
    iteration: int
    sub_queries_generated: int
    search_results_found: int
    documents_extracted: int
    sections_drafted: int
    avg_confidence: float
    sections_by_confidence: dict[str, int]
    token_usage: list[TokenUsage]
    replan_reason: str | None
    section_snapshots: list[SectionConfidenceSnapshot]


# --- Protocols ---


@runtime_checkable
class SearchBackend(Protocol):
    name: str

    async def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        category: str | None = None,
    ) -> list[SearchResult]: ...

    async def health_check(self) -> bool: ...


# --- Fan-out Input Types ---


class SearcherInput(TypedDict):
    sub_query: SubQuery


class ExtractorInput(TypedDict):
    search_result: SearchResult

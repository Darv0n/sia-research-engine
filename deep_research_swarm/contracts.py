"""Single source of truth for all types, enums, and protocols."""

from __future__ import annotations

from enum import Enum
from typing import NotRequired, Protocol, TypedDict, runtime_checkable

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


class ProvenanceRecord(TypedDict):
    """Tracks the full provenance chain for a piece of content."""

    entity_id: str  # URN-style, e.g. "urn:doi:10.1234/example"
    source_url: str
    source_kind: str  # "web" | "scholarly" | "archive" | "pdf"
    fetched_at: str  # ISO 8601
    extractor: str  # "crawl4ai" | "trafilatura" | "pymupdf4llm" | "wayback"
    license_tag: str  # "unknown" | "CC-BY-4.0" | etc.
    capture_timestamp: str  # Wayback-only, YYYYMMDDHHMMSS; "" for non-archive
    content_hash: str  # SHA-256 of extracted text (OE2: enables change detection)


class ScholarlyMetadata(TypedDict):
    """Metadata from academic APIs (OpenAlex, Semantic Scholar, Crossref)."""

    doi: str
    arxiv_id: str
    pmid: str
    title: str
    authors: list[str]
    year: int
    venue: str
    citation_count: int
    reference_count: int
    is_open_access: bool
    open_access_url: str
    abstract: str


class ArchiveCapture(TypedDict):
    """Wayback Machine snapshot data."""

    original_url: str
    archive_url: str
    capture_timestamp: str  # YYYYMMDDHHMMSS format
    status_code: int
    content_type: str


class SourcePassage(TypedDict):
    """A passage chunk from an extracted document.

    INVARIANT: SourcePassage.id MUST be content-hash-based (D1, I1).
    This enables: incremental research (diff runs), claim graph traversal,
    cross-run dedup. Do NOT change to random UUIDs. This is the load-bearing
    decision for E2 (incremental research).
    """

    id: str  # f"sp-{sha256((source_id + str(position)).encode()).hexdigest()[:8]}"
    source_id: str  # canonical_url + content_hash — CONTENT-STABLE, not run-stable
    source_url: str
    content: str  # 200-400 tokens target
    position: int  # 0-indexed chunk position
    char_offset: int
    token_count: int  # word_count * 1.3 estimate
    heading_context: str  # nearest heading above this passage
    claim_ids: list[str]  # OE1: empty in V7, populated in V8


class SectionOutline(TypedDict):
    """Outline structure for a section before drafting."""

    heading: str
    key_claims: list[str]
    source_ids: list[str]  # passage source_ids relevant to this section
    narrative_role: str  # "introduction" | "evidence" | "analysis" | "conclusion"


class IndexedContent(TypedDict):
    """V8 stub for embedding-based retrieval (OE8/D8).

    Ships empty in V7. V8 populates embedding + index_backend
    without surgery to contracts.py.
    """

    passage_id: str
    embedding: list[float]  # empty in V7
    index_backend: str  # empty in V7


class SearchResult(TypedDict):
    id: str
    sub_query_id: str
    url: str
    title: str
    snippet: str
    backend: str  # "searxng" | "exa" | "tavily" | "openalex" | "semantic_scholar"
    rank: int
    score: float  # 0-1
    authority: SourceAuthority
    timestamp: str  # ISO 8601
    provenance: NotRequired[ProvenanceRecord | None]
    scholarly_metadata: NotRequired[ScholarlyMetadata | None]


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
    grounding_score: NotRequired[float]  # OE3: 0-1, from verify_grounding()
    claim_details: NotRequired[list[dict]]  # OE3: per-claim grounding details


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


class ResearchMemory(TypedDict):
    thread_id: str
    question: str
    timestamp: str  # ISO 8601
    key_findings: list[str]  # section headings from section_drafts
    gaps: list[str]  # from research_gaps descriptions
    sources_count: int
    iterations: int
    converged: bool


class RunEvent(TypedDict):
    node: str
    iteration: int
    ts: str  # ISO 8601
    elapsed_s: float
    inputs_summary: dict[str, int]  # field -> count/size
    outputs_summary: dict[str, int]  # field -> count/size
    tokens: int
    cost: float


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


# --- Adaptive Control (V8) ---


class Tunable(TypedDict):
    """Definition of a single tunable parameter with bounded range."""

    name: str
    default: int | float
    floor: int | float
    ceiling: int | float
    category: str  # "extraction" | "scoring" | "grounding" | "search" | "synthesis"


class ComplexityProfile(TypedDict):
    """Observed pipeline metrics driving adaptive decisions."""

    result_count: int
    backends_used: int
    iteration: int
    extraction_success_rate: float  # 0-1
    mean_grounding_score: float  # 0-1
    token_spend_rate: float  # tokens_used / token_budget
    scored_doc_count: int
    citation_chain_yield: int  # papers found in citation chain
    volume_factor: float  # log10 scaling of result count
    backend_factor: float  # backend diversity scaling
    iter_factor: float  # iteration progression scaling
    multiplier: float  # final composite multiplier (0.5-2.0)


class AdaptationEvent(TypedDict):
    """Record of a single adaptive adjustment made by the overseer."""

    tunable_name: str
    old_value: int | float
    new_value: int | float
    reason: str  # human-readable explanation
    trigger: str  # "adapt_extraction" | "adapt_synthesis"
    iteration: int


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

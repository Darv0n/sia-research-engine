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


# --- SIA Types (V10) ---


class EntropyBand(str, Enum):
    """Thermodynamic entropy bands for reactor steering."""

    CRYSTALLINE = "crystalline"  # <= 0.20 — synthesis zone
    CONVERGENCE = "convergence"  # 0.20-0.45 — build and compress
    TURBULENCE = "turbulence"  # 0.45-0.70 — productive tension
    RUNAWAY = "runaway"  # > 0.70 — containment needed


class IntType(str, Enum):
    """Interaction type grammar for turn-to-turn dynamics."""

    B = "B"  # Build — expands on prior
    C = "C"  # Challenge — questions/refutes
    RF = "RF"  # Reframe — changes perspective
    CL = "CL"  # Clarify — requests clarification
    CO = "CO"  # Coalition — attempts alignment
    A = "A"  # Agreement — agrees with prior
    S = "S"  # Support — offers supporting evidence
    INTERRUPT = "I"  # Interrupt — breaks flow
    INIT = "INIT"  # Initial turn


class EntropyState(TypedDict):
    """Thermodynamic entropy measurement from ResearchState observables."""

    e: float  # scalar entropy in [0, 1]
    e_amb: float  # ambiguity component
    e_conf: float  # conflict component
    e_nov: float  # novelty component
    e_trust: float  # trust/coherence component
    band: str  # EntropyBand value
    turn: int
    stagnation_count: int  # consecutive turns with |delta_e| < 0.03
    _prev_query_count: NotRequired[int]  # for novelty delta computation
    _prev_headings: NotRequired[list[str]]  # for novelty heading delta


class TurnRecord(TypedDict):
    """Record of a single reactor deliberation turn."""

    turn: int
    agent: str  # SIAAgent.id
    int_type: str  # IntType value
    constraints: list[str]
    challenges: list[str]
    reframes: list[str]
    response_to_prior: list[str]
    raw_output: str


class ReactorState(TypedDict):
    """Accumulated state of the SIA reactor during deliberation."""

    constraints: list[str]
    rejected_branches: list[str]
    active_frames: list[str]
    key_claims: list[str]
    coalition_map: dict[str, list[str]]  # agent_id -> aligned agent_ids
    unresolved: list[str]
    turn_log: list[TurnRecord]


class ReactorTrace(TypedDict):
    """Post-hoc trace of a reactor run for diagnostics."""

    turns_executed: int
    agents_used: list[str]
    constraints_produced: int
    branches_killed: int
    challenges_issued: int
    final_entropy: float
    termination_reason: str
    ignition_pattern: str


class AdversarialFinding(TypedDict):
    """A single finding from adversarial critique."""

    agent: str
    int_type: str  # IntType value
    target_section: str  # section_id or "global"
    finding: str
    severity: str  # "critical" | "significant" | "minor"
    actionable: bool
    response_to: str


class CritiqueTrace(TypedDict):
    """Post-hoc trace of adversarial critique."""

    turns: int
    findings_count: int
    critical_findings: int
    constraints_extracted: int
    missing_variables: list[str]
    alternative_frames: list[str]
    recommendation: str  # "converge" | "replan" | "refine_targeted"


# --- Tensegrity Types (V10) ---


class Facet(TypedDict):
    """A decomposed sub-question from the research question."""

    id: str  # "facet-001"
    question: str
    weight: float  # importance 0.0-1.0


class ClaimVerdict(TypedDict):
    """Per-claim merged judgment from deliberation panel."""

    claim_id: str
    claim_text: str
    grounding_score: float
    grounding_method: str  # "jaccard_v1" | "embedding_v1" | "nli_v1"
    authority_score: float
    authority_level: str  # SourceAuthority value
    contradicted: bool
    contradiction_id: NotRequired[str]


class ActiveTension(TypedDict):
    """A contradiction annotated with authority and grounding context."""

    id: str
    claim_a: ClaimVerdict
    claim_b: ClaimVerdict
    severity: str  # "direct" | "nuanced" | "contextual"
    authority_differential: float
    resolution_hint: str


class CoverageMap(TypedDict):
    """Coverage assessment across research facets."""

    facet_coverage: dict[str, float]  # facet_id -> coverage 0.0-1.0
    overall_coverage: float
    uncovered_facets: list[str]
    under_represented_perspectives: list[str]


class AuthorityProfile(TypedDict):
    """Authority distribution summary for a passage cluster."""

    dominant_authority: str  # SourceAuthority value
    source_count: int
    avg_authority_score: float
    institutional_ratio: float


class PassageCluster(TypedDict):
    """A group of related passages clustered by embedding similarity."""

    cluster_id: str
    theme: str
    passage_ids: list[str]
    claims: list[ClaimVerdict]
    authority: AuthorityProfile
    summary: str  # Haiku-generated


class CrossClusterTension(TypedDict):
    """A tension that spans two passage clusters."""

    cluster_a_id: str
    cluster_b_id: str
    tension: ActiveTension


class JudgmentContext(TypedDict):
    """Merged output from all four deliberation panel agents."""

    claim_verdicts: list[ClaimVerdict]
    source_credibility: dict[str, float]  # URL -> merged credibility
    active_tensions: list[ActiveTension]
    coverage_map: CoverageMap
    next_wave_queries: list[SubQuery]
    overall_coverage: float
    structural_risks: list[str]
    wave_number: int
    facets: list[Facet]
    passage_to_claims: NotRequired[dict[str, list[str]]]  # pid -> [claim_id]


class KnowledgeArtifact(TypedDict):
    """Pre-deliberated, pre-verified, pre-ranked knowledge for synthesis."""

    question: str
    facets: list[Facet]
    clusters: list[PassageCluster]
    claim_verdicts: list[ClaimVerdict]
    active_tensions: list[ActiveTension]
    coverage: CoverageMap
    insights: list[dict]  # flexible insight structures
    authority_profiles: list[AuthorityProfile]
    structural_risks: list[str]
    compression_ratio: float  # claims_out / passages_in (information density)
    wave_count: int


class SwarmMetadata(TypedDict):
    """Metadata from multi-reactor swarm execution."""

    n_reactors: int
    reactor_configs: list[dict]
    reactor_entropies: list[float]
    reactor_tokens: list[int]
    reactor_costs: list[float]
    winner_id: str
    selection_reason: str
    selection_scores: dict[str, float]  # reactor_id -> composite score
    cross_validation_scores: dict[str, float]  # reactor_id -> cross-val score
    total_tokens_all: int
    total_cost_all: float
    failed_reactors: list[str]

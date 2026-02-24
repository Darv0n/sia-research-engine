"""Citation chaining node — BFS traversal of scholarly citation graphs (V7, PR-08).

Expands evidence by discovering papers reachable through references and citations
of top-scored scholarly documents. Uses Semantic Scholar get_paper_details() for
graph traversal.

Pipeline: score -> citation_chain -> contradiction

No-op pass-through when S2 backend unavailable. Zero impact on non-scholarly runs.
"""

from __future__ import annotations

import heapq
import uuid
from datetime import datetime, timezone

from deep_research_swarm.contracts import (
    ProvenanceRecord,
    ScholarlyMetadata,
    SearchResult,
    SourceAuthority,
)
from deep_research_swarm.graph.state import ResearchState
from deep_research_swarm.utils.text import jaccard_score

_BUDGET = 50  # Max total papers to process
_MAX_HOPS = 2  # BFS depth limit
_TOP_SEEDS = 5  # Number of seed papers from scored_documents
_CANDIDATES_PER_HOP = 20  # Max candidates to fetch details for per hop
_RELEVANCE_THRESHOLD = 0.15  # Minimum title relevance to consider


def _relevance_score(title: str, abstract: str, question: str) -> float:
    """Jaccard similarity between paper text and research question.

    Uses title when abstract unavailable (reference lists often lack abstracts).
    When both available, weighted combination: 0.4 title + 0.6 abstract.
    """
    if not question:
        return 0.0

    title_score = jaccard_score(title, question) if title else 0.0

    if abstract:
        abstract_score = jaccard_score(abstract, question)
        return 0.4 * title_score + 0.6 * abstract_score

    return title_score


def _extract_paper_id(result: SearchResult) -> str | None:
    """Extract Semantic Scholar paper ID from a SearchResult."""
    sm = result.get("scholarly_metadata")
    if not sm:
        return None

    # Try DOI first (most reliable for S2 lookup)
    doi = sm.get("doi", "")
    if doi:
        return doi

    # Try arXiv ID
    arxiv = sm.get("arxiv_id", "")
    if arxiv:
        return f"ArXiv:{arxiv}"

    # Try extracting S2 paper ID from URL
    url = result.get("url", "")
    if "semanticscholar.org/paper/" in url:
        # URL format: https://www.semanticscholar.org/paper/{paperId}
        parts = url.rstrip("/").split("/")
        return parts[-1] if parts else None

    return None


def _has_scholarly_results(state: ResearchState) -> bool:
    """Check if there are scored documents with scholarly metadata."""
    for doc in state.get("scored_documents", []):
        if doc.get("scholarly_metadata"):
            return True
    return False


def _build_search_result(paper: dict, question: str) -> SearchResult | None:
    """Build a SearchResult from a Semantic Scholar paper dict."""
    paper_id = paper.get("paperId", "")
    title = paper.get("title", "") or ""
    if not title:
        return None

    ext_ids = paper.get("externalIds", {}) or {}
    oa_pdf = paper.get("openAccessPdf", {}) or {}
    url = paper.get("url", "") or f"https://www.semanticscholar.org/paper/{paper_id}"

    authors = []
    for a in paper.get("authors", []):
        name = a.get("name", "")
        if name:
            authors.append(name)

    abstract = paper.get("abstract", "") or ""
    now = datetime.now(timezone.utc).isoformat()

    scholarly = ScholarlyMetadata(
        doi=ext_ids.get("DOI", "") or "",
        arxiv_id=ext_ids.get("ArXiv", "") or "",
        pmid=ext_ids.get("PubMed", "") or "",
        title=title,
        authors=authors,
        year=paper.get("year", 0) or 0,
        venue=paper.get("venue", "") or "",
        citation_count=paper.get("citationCount", 0) or 0,
        reference_count=paper.get("referenceCount", 0) or 0,
        is_open_access=paper.get("isOpenAccess", False) or False,
        open_access_url=oa_pdf.get("url", "") or "",
        abstract=abstract,
    )

    provenance = ProvenanceRecord(
        entity_id=f"urn:s2:{paper_id}",
        source_url=url,
        source_kind="scholarly",
        fetched_at=now,
        extractor="citation_chain",
        license_tag="unknown",
        capture_timestamp="",
        content_hash="",
    )

    return SearchResult(
        id=f"sr-{uuid.uuid4().hex[:8]}",
        sub_query_id="",
        url=url,
        title=title,
        snippet=abstract[:300] if abstract else title,
        backend="citation_chain",
        rank=0,
        score=0.0,
        authority=SourceAuthority.INSTITUTIONAL,
        timestamp=now,
        provenance=provenance,
        scholarly_metadata=scholarly,
    )


async def citation_chain(
    state: ResearchState,
    s2_backend,
) -> dict:
    """Expand evidence via citation graph BFS. No-op if no S2 backend."""
    if not s2_backend or not _has_scholarly_results(state):
        return {}

    # Read adaptive tunables (V8) — fall back to module-level defaults
    _snap = state.get("tunable_snapshot", {})
    budget = int(_snap.get("citation_chain_budget", _BUDGET))
    max_hops = int(_snap.get("citation_chain_max_hops", _MAX_HOPS))
    top_seeds = int(_snap.get("citation_chain_top_seeds", _TOP_SEEDS))

    question = state.get("research_question", "")

    # Collect seed paper IDs from top scored documents
    seeds: list[tuple[float, str]] = []
    for doc in state.get("scored_documents", []):
        sm = doc.get("scholarly_metadata")
        if not sm:
            continue
        doi = sm.get("doi", "")
        paper_id = doi if doi else None
        if not paper_id:
            arxiv = sm.get("arxiv_id", "")
            if arxiv:
                paper_id = f"ArXiv:{arxiv}"
        if not paper_id:
            url = doc.get("url", "")
            if "semanticscholar.org/paper/" in url:
                paper_id = url.rstrip("/").split("/")[-1]
        if paper_id:
            seeds.append((doc.get("combined_score", 0.0), paper_id))

    # Take top N seeds by combined_score
    seeds.sort(key=lambda x: x[0], reverse=True)
    seed_ids = [pid for _, pid in seeds[:top_seeds]]

    if not seed_ids:
        return {}

    # Build the seen set from previous citation chain results (cross-iteration dedup)
    seen: set[str] = set()
    for result in state.get("citation_chain_results", []):
        pid = _extract_paper_id(result)
        if pid:
            seen.add(pid)

    # Also add current scored documents
    for doc in state.get("scored_documents", []):
        sm = doc.get("scholarly_metadata")
        if sm:
            doi = sm.get("doi", "")
            if doi:
                seen.add(doi)

    # Add seed IDs to seen
    for sid in seed_ids:
        seen.add(sid)

    # BFS traversal
    frontier = list(seed_ids)
    new_results: list[SearchResult] = []

    for hop in range(max_hops):
        if len(seen) >= budget:
            break

        # Collect all candidate references/citations from frontier papers
        # Use a priority queue: (-score, paper_id, title)
        candidates: list[tuple[float, str, str]] = []

        for paper_id in frontier:
            if len(seen) >= budget:
                break

            details = await s2_backend.get_paper_details(paper_id)
            if not details:
                continue

            # Score references by title relevance (cheap, no API call)
            for ref in details.get("references", []) or []:
                ref_id = ref.get("paperId", "")
                ref_title = ref.get("title", "") or ""
                if not ref_id or ref_id in seen:
                    continue
                score = _relevance_score(ref_title, "", question)
                if score >= _RELEVANCE_THRESHOLD:
                    heapq.heappush(candidates, (-score, ref_id, ref_title))

            # Also check citations (papers that cite this one)
            for cit in details.get("citations", []) or []:
                cit_id = cit.get("paperId", "")
                cit_title = cit.get("title", "") or ""
                if not cit_id or cit_id in seen:
                    continue
                score = _relevance_score(cit_title, "", question)
                if score >= _RELEVANCE_THRESHOLD:
                    heapq.heappush(candidates, (-score, cit_id, cit_title))

        # Select top candidates to fetch details for
        next_frontier: list[str] = []
        fetched = 0

        while candidates and fetched < _CANDIDATES_PER_HOP:
            if len(seen) >= budget:
                break

            neg_score, cand_id, cand_title = heapq.heappop(candidates)
            if cand_id in seen:
                continue

            seen.add(cand_id)

            # Fetch full details for this candidate
            cand_details = await s2_backend.get_paper_details(cand_id)
            if not cand_details:
                continue

            # Re-score with abstract for final selection
            abstract = cand_details.get("abstract", "") or ""
            final_score = _relevance_score(cand_details.get("title", "") or "", abstract, question)

            if final_score >= _RELEVANCE_THRESHOLD:
                sr = _build_search_result(cand_details, question)
                if sr:
                    new_results.append(sr)
                    next_frontier.append(cand_id)

            fetched += 1

        frontier = next_frontier

    if not new_results:
        return {}

    return {
        "citation_chain_results": new_results,
        "search_results": new_results,
    }

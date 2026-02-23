"""Crossref/Unpaywall DOI resolution utilities (V7, PR-04).

NOT a SearchBackend — these are utility functions for DOI resolution,
open access URL discovery, and scholarly result enrichment.
Used by other backends and the extractor to enrich results with
canonical metadata.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from deep_research_swarm.contracts import ScholarlyMetadata, SearchResult

_CROSSREF_BASE = "https://api.crossref.org"
_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_MAX_RETRIES = 2
_TIMEOUT = 15.0

# DOI regex: 10.XXXX/anything
_DOI_RE = re.compile(r"10\.\d{4,}/[^\s]+")


def normalize_doi(doi: str) -> str:
    """Normalize a DOI string to bare form (no URL prefix).

    Handles:
    - https://doi.org/10.1234/example -> 10.1234/example
    - http://dx.doi.org/10.1234/example -> 10.1234/example
    - doi:10.1234/example -> 10.1234/example
    - 10.1234/example -> 10.1234/example (already bare)
    """
    doi = doi.strip()
    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )
    for prefix in prefixes:
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix) :]
            break
    return doi.strip()


def extract_doi_from_url(url: str) -> str | None:
    """Extract a DOI from a URL if present."""
    match = _DOI_RE.search(url)
    return match.group(0) if match else None


async def resolve_doi(
    doi: str,
    *,
    email: str = "",
) -> dict | None:
    """Resolve a DOI via Crossref API, returning work metadata.

    Returns the Crossref work message dict, or None on failure.
    """
    doi = normalize_doi(doi)
    headers: dict[str, str] = {"User-Agent": "deep-research-swarm/0.7"}
    if email:
        headers["From"] = email

    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.get(f"{_CROSSREF_BASE}/works/{doi}")
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data.get("message")
            except (httpx.HTTPError, ValueError):
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(1.0)
                    continue
                return None
    return None


async def find_open_access_url(
    doi: str,
    *,
    email: str = "",
) -> str | None:
    """Find an open access URL for a DOI via Unpaywall.

    Returns the best open access URL, or None if not available.
    Requires an email for Unpaywall API access.
    """
    if not email:
        return None

    doi = normalize_doi(doi)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.get(
                f"{_UNPAYWALL_BASE}/{doi}",
                params={"email": email},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            best_location = data.get("best_oa_location", {}) or {}
            return best_location.get("url_for_pdf") or best_location.get("url") or None
        except (httpx.HTTPError, ValueError):
            return None


async def doi_content_negotiate(
    doi: str,
    *,
    accept: str = "application/vnd.citationstyles.csl+json",
) -> dict | None:
    """Content-negotiate a DOI for structured metadata.

    By default requests CSL-JSON format. Returns parsed JSON or None.
    """
    doi = normalize_doi(doi)
    url = f"https://doi.org/{doi}"
    headers = {"Accept": accept, "User-Agent": "deep-research-swarm/0.7"}

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return None


def _crossref_to_scholarly(work: dict) -> ScholarlyMetadata:
    """Convert Crossref work metadata to ScholarlyMetadata."""
    doi = work.get("DOI", "")

    # Authors
    authors: list[str] = []
    for author in work.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    # Year from published-print or published-online
    year = 0
    for date_field in ("published-print", "published-online", "created"):
        date_parts = work.get(date_field, {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
            break

    # Venue from container-title
    container = work.get("container-title", [])
    venue = container[0] if container else ""

    # Open access: Crossref doesn't directly provide this, but we can check license
    licenses = work.get("license", [])
    is_oa = any("creativecommons" in (lic.get("URL", "") or "").lower() for lic in licenses)

    return ScholarlyMetadata(
        doi=doi,
        arxiv_id="",
        pmid="",
        title=work.get("title", [""])[0] if work.get("title") else "",
        authors=authors,
        year=year,
        venue=venue,
        citation_count=work.get("is-referenced-by-count", 0) or 0,
        reference_count=work.get("references-count", 0) or 0,
        is_open_access=is_oa,
        open_access_url="",
        abstract=work.get("abstract", "") or "",
    )


async def enrich_scholarly_result(
    result: SearchResult,
    *,
    email: str = "",
) -> SearchResult:
    """Enrich a SearchResult with Crossref metadata if it has a DOI.

    If the result already has scholarly_metadata, returns unchanged.
    If a DOI can be extracted from the URL, resolves via Crossref.
    """
    # Skip if already enriched
    if result.get("scholarly_metadata"):
        return result

    # Try to extract DOI from URL
    doi = extract_doi_from_url(result["url"])
    if not doi:
        return result

    work = await resolve_doi(doi, email=email)
    if not work:
        return result

    scholarly = _crossref_to_scholarly(work)

    # Find OA URL if email provided
    oa_url = await find_open_access_url(doi, email=email)
    if oa_url and not scholarly["open_access_url"]:
        scholarly = {**scholarly, "open_access_url": oa_url, "is_open_access": True}

    return {**result, "scholarly_metadata": scholarly}

"""Source diversity scoring using the Herfindahl-Hirschman Index (HHI)."""

from __future__ import annotations

from urllib.parse import urlparse

from deep_research_swarm.contracts import DiversityMetrics, ScoredDocument, SourceAuthority


def _extract_domain(url: str) -> str:
    """Extract the registered domain from a URL, lowercased."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Strip 'www.' prefix for normalization
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower()
    except Exception:
        return "unknown"


def compute_diversity(documents: list[ScoredDocument]) -> DiversityMetrics:
    """Compute source diversity metrics for a set of scored documents.

    Uses the Herfindahl-Hirschman Index (HHI) to measure domain concentration.
    HHI = sum of squared market shares. Ranges from 1/N (uniform) to 1 (monopoly).
    Diversity score is (1 - HHI), normalized so 0 = worst, 1 = best.
    """
    if not documents:
        return DiversityMetrics(
            unique_domains=0,
            total_documents=0,
            domain_concentration=0.0,
            authority_distribution={},
            diversity_score=0.0,
        )

    # Count documents per domain
    domain_counts: dict[str, int] = {}
    authority_counts: dict[str, int] = {}

    for doc in documents:
        domain = _extract_domain(doc["url"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        auth = doc["authority"]
        auth_key = auth.value if isinstance(auth, SourceAuthority) else str(auth)
        authority_counts[auth_key] = authority_counts.get(auth_key, 0) + 1

    total = len(documents)
    unique = len(domain_counts)

    # Compute HHI: sum of squared market shares
    hhi = sum((count / total) ** 2 for count in domain_counts.values())

    # Normalize diversity score: 1 - HHI gives 0 (monopoly) to ~1 (uniform)
    diversity_score = round(1.0 - hhi, 4)

    return DiversityMetrics(
        unique_domains=unique,
        total_documents=total,
        domain_concentration=round(hhi, 4),
        authority_distribution=authority_counts,
        diversity_score=diversity_score,
    )

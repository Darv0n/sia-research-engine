"""Settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load .env from project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()


@dataclass(frozen=True)
class Settings:
    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    exa_api_key: str = field(default_factory=lambda: os.environ.get("EXA_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))

    # SearXNG
    searxng_url: str = field(
        default_factory=lambda: os.environ.get("SEARXNG_URL", "http://localhost:8080")
    )

    # Models
    opus_model: str = field(default_factory=lambda: os.environ.get("OPUS_MODEL", "claude-opus-4-6"))
    sonnet_model: str = field(
        default_factory=lambda: os.environ.get("SONNET_MODEL", "claude-sonnet-4-6")
    )
    haiku_model: str = field(
        default_factory=lambda: os.environ.get("HAIKU_MODEL", "claude-haiku-4-5-20251001")
    )

    # Limits
    max_iterations: int = field(default_factory=lambda: int(os.environ.get("MAX_ITERATIONS", "3")))
    token_budget: int = field(default_factory=lambda: int(os.environ.get("TOKEN_BUDGET", "200000")))
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_REQUESTS", "5"))
    )

    # Scoring
    authority_weight: float = field(
        default_factory=lambda: float(os.environ.get("AUTHORITY_WEIGHT", "0.2"))
    )
    rrf_k: int = field(default_factory=lambda: int(os.environ.get("RRF_K", "60")))

    # Convergence
    convergence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONVERGENCE_THRESHOLD", "0.05"))
    )

    # Search cache
    search_cache_ttl: int = field(
        default_factory=lambda: int(os.environ.get("SEARCH_CACHE_TTL", "3600"))
    )
    search_cache_dir: str = field(
        default_factory=lambda: os.environ.get("SEARCH_CACHE_DIR", ".cache/search")
    )

    # Checkpointing (V4)
    checkpoint_db: str = field(
        default_factory=lambda: os.environ.get("CHECKPOINT_DB", "checkpoints/research.db")
    )
    checkpoint_backend: str = field(
        default_factory=lambda: os.environ.get("CHECKPOINT_BACKEND", "sqlite")
    )

    # Memory (V5)
    memory_dir: str = field(default_factory=lambda: os.environ.get("MEMORY_DIR", "memory/"))

    # PostgresSaver (V5)
    postgres_dsn: str = field(default_factory=lambda: os.environ.get("POSTGRES_DSN", ""))

    # Run event log (V6)
    run_log_dir: str = field(default_factory=lambda: os.environ.get("RUN_LOG_DIR", "runs/"))

    # Execution mode (V6)
    mode: str = field(default_factory=lambda: os.environ.get("MODE", "auto"))

    # Scholarly backends (V7)
    openalex_email: str = field(default_factory=lambda: os.environ.get("OPENALEX_EMAIL", ""))
    openalex_api_key: str = field(default_factory=lambda: os.environ.get("OPENALEX_API_KEY", ""))
    semantic_scholar_api_key: str = field(default_factory=lambda: os.environ.get("S2_API_KEY", ""))

    # Archive backends (V7)
    wayback_enabled: bool = field(
        default_factory=lambda: os.environ.get("WAYBACK_ENABLED", "true").lower() == "true"
    )
    wayback_timeout: int = field(
        default_factory=lambda: int(os.environ.get("WAYBACK_TIMEOUT", "15"))
    )

    # Adaptive control (V8)
    adaptive_mode: bool = field(
        default_factory=lambda: os.environ.get("ADAPTIVE_MODE", "true").lower() == "true"
    )

    # Embedding model (V8, optional — only used when fastembed installed)
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # GROBID server (V8, optional — for structured PDF extraction)
    grobid_url: str = field(default_factory=lambda: os.environ.get("GROBID_URL", ""))

    # SIA (V10) — multi-agent deliberation
    sia_enabled: bool = field(
        default_factory=lambda: os.environ.get("SIA_ENABLED", "true").lower() == "true"
    )

    # Swarm (V10) — multi-reactor orchestration
    swarm_enabled: bool = field(
        default_factory=lambda: os.environ.get("SWARM_ENABLED", "true").lower() == "true"
    )
    swarm_max_reactors: int = field(
        default_factory=lambda: int(os.environ.get("SWARM_MAX_REACTORS", "5"))
    )

    def available_backends(self) -> list[str]:
        """Return list of backends that have valid configuration."""
        backends = ["searxng"]  # Always available (local)
        if self.exa_api_key:
            backends.append("exa")
        if self.tavily_api_key:
            backends.append("tavily")
        # Scholarly backends (V7)
        if self.openalex_email:
            backends.append("openalex")
        backends.append("semantic_scholar")  # Works unauthenticated
        if self.wayback_enabled:
            backends.append("wayback")
        return backends

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        if self.max_iterations < 1:
            errors.append("MAX_ITERATIONS must be >= 1")
        if self.token_budget < 1000:
            errors.append("TOKEN_BUDGET must be >= 1000")
        if self.checkpoint_backend not in ("sqlite", "none", "postgres"):
            errors.append(
                f"CHECKPOINT_BACKEND must be 'sqlite', 'postgres', or 'none',"
                f" got '{self.checkpoint_backend}'"
            )
        if self.checkpoint_backend == "postgres" and not self.postgres_dsn:
            errors.append("POSTGRES_DSN is required when CHECKPOINT_BACKEND is 'postgres'")
        if self.mode not in ("auto", "hitl"):
            errors.append(f"MODE must be 'auto' or 'hitl', got '{self.mode}'")
        if self.swarm_max_reactors < 2 or self.swarm_max_reactors > 10:
            errors.append(f"SWARM_MAX_REACTORS must be 2-10, got {self.swarm_max_reactors}")
        return errors

    def warnings(self) -> list[str]:
        """Return list of non-fatal configuration warnings."""
        warns: list[str] = []
        if self.openalex_email:
            # OpenAlex is configured — no warning needed
            pass
        elif "openalex" not in self.available_backends():
            # Not configured at all — no warning needed
            pass
        # Note: openalex appears in available_backends() only with email, so
        # the anonymous-pool case is: email empty but user explicitly requested it.

        if self.wayback_enabled and self.wayback_timeout < 5:
            warns.append(
                f"WAYBACK_TIMEOUT={self.wayback_timeout}s is aggressive. "
                "Wayback CDX responses can be slow — consider >= 15s."
            )
        return warns


def get_settings() -> Settings:
    """Create Settings from current environment."""
    return Settings()

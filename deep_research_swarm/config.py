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

    def available_backends(self) -> list[str]:
        """Return list of backends that have valid configuration."""
        backends = ["searxng"]  # Always available (local)
        if self.exa_api_key:
            backends.append("exa")
        if self.tavily_api_key:
            backends.append("tavily")
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
        return errors


def get_settings() -> Settings:
    """Create Settings from current environment."""
    return Settings()

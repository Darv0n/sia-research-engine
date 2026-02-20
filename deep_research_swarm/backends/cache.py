"""File-based search cache keyed on (query, backend, num_results) with TTL."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class SearchCache:
    """File-based search result cache with SHA-256 keys and TTL expiry."""

    def __init__(self, cache_dir: str | Path, ttl: int = 3600) -> None:
        self._dir = Path(cache_dir)
        self._ttl = ttl
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(query: str, backend: str, num_results: int) -> str:
        """Deterministic SHA-256 key from normalized inputs."""
        raw = f"{query.strip().lower()}|{backend.strip().lower()}|{num_results}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, query: str, backend: str, num_results: int) -> list[dict] | None:
        """Return cached results or None on miss / expiry / corruption."""
        key = self._make_key(query, backend, num_results)
        path = self._path_for(key)

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupted file — treat as miss and clean up
            path.unlink(missing_ok=True)
            return None

        stored_at = data.get("stored_at", 0)
        if time.time() - stored_at > self._ttl:
            path.unlink(missing_ok=True)
            return None

        return data.get("results")

    def put(self, query: str, backend: str, num_results: int, results: list[dict]) -> None:
        """Store results with current timestamp."""
        key = self._make_key(query, backend, num_results)
        path = self._path_for(key)
        payload = {
            "stored_at": time.time(),
            "query": query,
            "backend": backend,
            "num_results": num_results,
            "results": results,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def clear(self) -> int:
        """Remove all cache files. Returns count of files removed."""
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        return count

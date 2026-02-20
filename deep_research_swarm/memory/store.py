"""File-based memory store for cross-session research persistence."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research_swarm.contracts import ResearchMemory
from deep_research_swarm.utils.text import jaccard_score


class MemoryStore:
    """JSON-backed store for research memory records.

    Pattern follows backends/cache.py: single JSON file, human-readable,
    graceful degradation on missing/corrupt data.
    """

    _FILENAME = "research-memory.json"

    def __init__(self, memory_dir: str | Path) -> None:
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _path(self) -> Path:
        return self._dir / self._FILENAME

    def _load(self) -> list[ResearchMemory]:
        """Load all memory records. Returns [] on any error."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, records: list[ResearchMemory]) -> None:
        """Write records to disk."""
        self._path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_record(self, record: ResearchMemory) -> None:
        """Append a new memory record."""
        records = self._load()
        records.append(record)
        self._save(records)

    def search(self, question: str, top_k: int = 3, min_score: float = 0.2) -> list[ResearchMemory]:
        """Find the most relevant prior research by Jaccard similarity on questions.

        Returns up to top_k records with score >= min_score, sorted descending.
        """
        records = self._load()
        if not records:
            return []

        scored: list[tuple[float, ResearchMemory]] = []
        for record in records:
            score = jaccard_score(question, record.get("question", ""))
            if score >= min_score:
                scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def list_all(self) -> list[ResearchMemory]:
        """Return all stored memory records."""
        return self._load()

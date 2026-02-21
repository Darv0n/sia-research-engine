"""Append-only JSONL event log for run observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from deep_research_swarm.contracts import RunEvent


class EventLog:
    """JSONL-backed event log for a single research run.

    Pattern follows MemoryStore/SearchCache: file-based, directory
    auto-creation, graceful degradation on read errors.
    """

    _FILENAME = "events.jsonl"

    def __init__(self, log_dir: str | Path, thread_id: str) -> None:
        self._dir = Path(log_dir) / thread_id
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._dir / self._FILENAME

    def emit(self, event: RunEvent) -> None:
        """Append a single event as a JSON line."""
        line = json.dumps(event, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self) -> list[RunEvent]:
        """Read all events. Skips corrupt lines, returns [] on missing file."""
        if not self.path.exists():
            return []
        events: list[RunEvent] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return events

    @staticmethod
    def make_event(
        *,
        node: str,
        iteration: int,
        elapsed_s: float,
        inputs_summary: dict[str, int] | None = None,
        outputs_summary: dict[str, int] | None = None,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> RunEvent:
        """Factory for creating a RunEvent with timestamp."""
        return RunEvent(
            node=node,
            iteration=iteration,
            ts=datetime.now(timezone.utc).isoformat(),
            elapsed_s=round(elapsed_s, 3),
            inputs_summary=inputs_summary or {},
            outputs_summary=outputs_summary or {},
            tokens=tokens,
            cost=round(cost, 6),
        )

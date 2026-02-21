"""Tests for event_log.writer — EventLog write/read, append semantics, graceful degradation."""

from __future__ import annotations

import json

from deep_research_swarm.event_log.writer import EventLog


def _make_event(**overrides) -> dict:
    base = {
        "node": "plan",
        "iteration": 1,
        "ts": "2026-02-21T12:00:00+00:00",
        "elapsed_s": 1.234,
        "inputs_summary": {"research_question": 1},
        "outputs_summary": {"sub_queries": 5},
        "tokens": 1500,
        "cost": 0.045,
    }
    base.update(overrides)
    return base


class TestEventLogWriteRead:
    def test_write_and_read_roundtrip(self, tmp_path):
        log = EventLog(tmp_path, "test-thread")
        event = _make_event()
        log.emit(event)
        events = log.read_all()
        assert len(events) == 1
        assert events[0]["node"] == "plan"
        assert events[0]["tokens"] == 1500

    def test_append_semantics(self, tmp_path):
        log = EventLog(tmp_path, "test-thread")
        log.emit(_make_event(node="plan"))
        log.emit(_make_event(node="search"))
        log.emit(_make_event(node="extract"))
        events = log.read_all()
        assert len(events) == 3
        assert [e["node"] for e in events] == ["plan", "search", "extract"]

    def test_empty_read(self, tmp_path):
        log = EventLog(tmp_path, "test-thread")
        assert log.read_all() == []

    def test_directory_auto_creation(self, tmp_path):
        log_dir = tmp_path / "deep" / "nested"
        log = EventLog(log_dir, "test-thread")
        log.emit(_make_event())
        assert log.path.exists()


class TestEventLogGracefulDegradation:
    def test_corrupt_jsonl_skips_bad_lines(self, tmp_path):
        log = EventLog(tmp_path, "test-thread")
        log.emit(_make_event(node="good"))
        # Inject a corrupt line
        with log.path.open("a", encoding="utf-8") as f:
            f.write("not valid json\n")
        log.emit(_make_event(node="also_good"))

        events = log.read_all()
        assert len(events) == 2
        assert events[0]["node"] == "good"
        assert events[1]["node"] == "also_good"

    def test_blank_lines_ignored(self, tmp_path):
        log = EventLog(tmp_path, "test-thread")
        log.emit(_make_event())
        with log.path.open("a", encoding="utf-8") as f:
            f.write("\n\n")
        events = log.read_all()
        assert len(events) == 1


class TestMakeEvent:
    def test_factory_produces_valid_event(self):
        event = EventLog.make_event(
            node="search",
            iteration=2,
            elapsed_s=3.14159,
            inputs_summary={"sub_queries": 4},
            outputs_summary={"search_results": 20},
            tokens=500,
            cost=0.015,
        )
        assert event["node"] == "search"
        assert event["iteration"] == 2
        assert event["elapsed_s"] == 3.142  # rounded to 3 decimals
        assert event["tokens"] == 500
        assert event["cost"] == 0.015
        assert "ts" in event

    def test_factory_defaults(self):
        event = EventLog.make_event(node="plan", iteration=1, elapsed_s=0.5)
        assert event["tokens"] == 0
        assert event["cost"] == 0.0
        assert event["inputs_summary"] == {}
        assert event["outputs_summary"] == {}

    def test_cost_rounding(self):
        event = EventLog.make_event(
            node="synthesize",
            iteration=1,
            elapsed_s=2.0,
            cost=0.1234567,
        )
        assert event["cost"] == 0.123457


class TestEventLogPath:
    def test_path_includes_thread_id(self, tmp_path):
        log = EventLog(tmp_path, "research-20260221-120000-abcd")
        assert "research-20260221-120000-abcd" in str(log.path)
        assert log.path.name == "events.jsonl"

    def test_jsonl_format(self, tmp_path):
        log = EventLog(tmp_path, "test-thread")
        log.emit(_make_event(node="a"))
        log.emit(_make_event(node="b"))
        lines = log.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "node" in parsed

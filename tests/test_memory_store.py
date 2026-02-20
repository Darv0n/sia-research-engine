"""Tests for memory.store — MemoryStore load/save/search with tmp_path."""

from __future__ import annotations

import json

from deep_research_swarm.memory.store import MemoryStore


def _make_record(**overrides) -> dict:
    base = {
        "thread_id": "research-20260220-120000-abcd",
        "question": "What is quantum computing?",
        "timestamp": "2026-02-20T12:00:00+00:00",
        "key_findings": ["Introduction to Qubits", "Quantum Gates"],
        "gaps": ["Error correction approaches"],
        "sources_count": 15,
        "iterations": 3,
        "converged": True,
    }
    base.update(overrides)
    return base


class TestMemoryStoreBasics:
    def test_empty_store_returns_empty_list(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        assert store.list_all() == []

    def test_add_and_list(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        rec = _make_record()
        store.add_record(rec)
        records = store.list_all()
        assert len(records) == 1
        assert records[0]["thread_id"] == rec["thread_id"]

    def test_multiple_records(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        store.add_record(_make_record(thread_id="t1"))
        store.add_record(_make_record(thread_id="t2"))
        assert len(store.list_all()) == 2

    def test_persistence_across_instances(self, tmp_path):
        mem_dir = tmp_path / "mem"
        store1 = MemoryStore(mem_dir)
        store1.add_record(_make_record())

        store2 = MemoryStore(mem_dir)
        assert len(store2.list_all()) == 1

    def test_creates_directory(self, tmp_path):
        mem_dir = tmp_path / "deep" / "nested" / "mem"
        store = MemoryStore(mem_dir)
        store.add_record(_make_record())
        assert mem_dir.exists()


class TestMemoryStoreSearch:
    def test_search_finds_exact_match(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        store.add_record(_make_record(question="What is quantum computing?"))
        results = store.search("What is quantum computing?")
        assert len(results) == 1

    def test_search_finds_similar(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        store.add_record(_make_record(question="quantum computing applications"))
        results = store.search("quantum computing breakthroughs")
        assert len(results) == 1

    def test_search_no_match(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        store.add_record(_make_record(question="quantum computing"))
        results = store.search("medieval history of France")
        assert len(results) == 0

    def test_search_respects_top_k(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        for i in range(5):
            store.add_record(
                _make_record(
                    thread_id=f"t{i}",
                    question=f"quantum computing topic {i}",
                )
            )
        results = store.search("quantum computing topic", top_k=2)
        assert len(results) == 2

    def test_search_empty_store(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        results = store.search("anything")
        assert results == []

    def test_search_respects_min_score(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        store.add_record(_make_record(question="completely unrelated bananas"))
        results = store.search("quantum computing", min_score=0.5)
        assert len(results) == 0


class TestMemoryStoreGracefulDegradation:
    def test_corrupt_json_returns_empty(self, tmp_path):
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "research-memory.json").write_text("not json!!!", encoding="utf-8")

        store = MemoryStore(mem_dir)
        assert store.list_all() == []

    def test_non_list_json_returns_empty(self, tmp_path):
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "research-memory.json").write_text(
            json.dumps({"not": "a list"}), encoding="utf-8"
        )

        store = MemoryStore(mem_dir)
        assert store.list_all() == []

    def test_add_to_corrupt_file_starts_fresh(self, tmp_path):
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "research-memory.json").write_text("corrupt!", encoding="utf-8")

        store = MemoryStore(mem_dir)
        store.add_record(_make_record())
        assert len(store.list_all()) == 1

"""Tests for memory.extract — extract_memory_record from graph state."""

from __future__ import annotations

from deep_research_swarm.memory.extract import extract_memory_record


class TestExtractMemoryRecord:
    def test_full_state(self):
        state = {
            "research_question": "What is quantum computing?",
            "section_drafts": [
                {"heading": "Introduction", "content": "..."},
                {"heading": "Applications", "content": "..."},
            ],
            "research_gaps": [
                {
                    "description": "Error correction",
                    "attempted_queries": [],
                    "reason": "low_confidence",
                },
            ],
            "search_results": [
                {"url": "https://a.com", "id": "1"},
                {"url": "https://b.com", "id": "2"},
                {"url": "https://a.com", "id": "3"},  # duplicate URL
            ],
            "iteration_history": [{"iteration": 1}, {"iteration": 2}],
            "converged": True,
        }
        record = extract_memory_record(state, "test-thread-001")

        assert record["thread_id"] == "test-thread-001"
        assert record["question"] == "What is quantum computing?"
        assert record["key_findings"] == ["Introduction", "Applications"]
        assert record["gaps"] == ["Error correction"]
        assert record["sources_count"] == 2  # deduplicated by URL
        assert record["iterations"] == 2
        assert record["converged"] is True
        assert "timestamp" in record

    def test_empty_state(self):
        record = extract_memory_record({}, "empty-thread")

        assert record["thread_id"] == "empty-thread"
        assert record["question"] == ""
        assert record["key_findings"] == []
        assert record["gaps"] == []
        assert record["sources_count"] == 0
        assert record["iterations"] == 0
        assert record["converged"] is False

    def test_missing_headings_skipped(self):
        state = {
            "section_drafts": [
                {"heading": "Valid", "content": "..."},
                {"heading": "", "content": "..."},  # empty heading
                {"content": "..."},  # no heading key
            ],
        }
        record = extract_memory_record(state, "t1")
        assert record["key_findings"] == ["Valid"]

    def test_missing_gap_descriptions_skipped(self):
        state = {
            "research_gaps": [
                {"description": "Real gap", "attempted_queries": [], "reason": "no_sources"},
                {"description": "", "attempted_queries": [], "reason": "no_sources"},
            ],
        }
        record = extract_memory_record(state, "t1")
        assert record["gaps"] == ["Real gap"]

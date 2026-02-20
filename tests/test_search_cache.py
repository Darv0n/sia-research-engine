"""Tests for SearchCache — file-based search result caching."""

from __future__ import annotations

import json
import time

import pytest

from deep_research_swarm.backends.cache import SearchCache


@pytest.fixture
def cache(tmp_path):
    return SearchCache(cache_dir=tmp_path / "cache", ttl=60)


@pytest.fixture
def sample_results():
    return [
        {"url": "https://example.com/1", "title": "Result 1"},
        {"url": "https://example.com/2", "title": "Result 2"},
    ]


class TestSearchCache:
    def test_miss_returns_none(self, cache):
        """Cache miss returns None."""
        assert cache.get("some query", "searxng", 10) is None

    def test_round_trip(self, cache, sample_results):
        """Put then get returns identical results."""
        cache.put("quantum physics", "searxng", 10, sample_results)
        got = cache.get("quantum physics", "searxng", 10)
        assert got == sample_results

    def test_ttl_expiry(self, tmp_path, sample_results):
        """Expired entries return None and are cleaned up."""
        cache = SearchCache(cache_dir=tmp_path / "cache", ttl=1)
        cache.put("query", "exa", 5, sample_results)

        # Backdate the stored_at
        key = SearchCache._make_key("query", "exa", 5)
        path = cache._path_for(key)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["stored_at"] = time.time() - 100
        path.write_text(json.dumps(data), encoding="utf-8")

        assert cache.get("query", "exa", 5) is None
        assert not path.exists()  # Cleaned up

    def test_key_determinism(self):
        """Same inputs produce same key across calls."""
        k1 = SearchCache._make_key("hello world", "searxng", 10)
        k2 = SearchCache._make_key("hello world", "searxng", 10)
        assert k1 == k2

    def test_case_insensitive_keys(self):
        """Keys are case-insensitive for query and backend."""
        k1 = SearchCache._make_key("Quantum Physics", "SearXNG", 10)
        k2 = SearchCache._make_key("quantum physics", "searxng", 10)
        assert k1 == k2

    def test_clear(self, cache, sample_results):
        """Clear removes all cached entries."""
        cache.put("q1", "searxng", 10, sample_results)
        cache.put("q2", "exa", 5, sample_results)
        removed = cache.clear()
        assert removed == 2
        assert cache.get("q1", "searxng", 10) is None
        assert cache.get("q2", "exa", 5) is None

    def test_corrupted_file(self, cache):
        """Corrupted JSON file treated as miss and cleaned up."""
        key = SearchCache._make_key("query", "searxng", 10)
        path = cache._path_for(key)
        path.write_text("not valid json{{{", encoding="utf-8")

        assert cache.get("query", "searxng", 10) is None
        assert not path.exists()

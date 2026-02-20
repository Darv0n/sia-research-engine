"""Tests for backend protocol conformance and registry."""

from deep_research_swarm.backends import available_backends, get_backend, register_backend
from deep_research_swarm.contracts import SearchBackend


class TestRegistry:
    def test_register_and_get(self):
        class DummyBackend:
            name = "dummy"

            def __init__(self, **kwargs):
                pass

            async def search(self, query, *, num_results=10, category=None):
                return []

            async def health_check(self):
                return True

        register_backend("dummy", DummyBackend)
        backend = get_backend("dummy")
        assert backend.name == "dummy"
        assert isinstance(backend, SearchBackend)

    def test_unknown_backend_raises(self):
        import pytest

        with pytest.raises(KeyError, match="Unknown backend"):
            get_backend("nonexistent_backend_xyz")

    def test_available_backends_returns_list(self):
        backends = available_backends()
        assert isinstance(backends, list)


class TestSearXNGConformance:
    def test_protocol_conformance(self):
        # Import to trigger registration
        from deep_research_swarm.backends.searxng import SearXNGBackend

        backend = SearXNGBackend(base_url="http://localhost:8080")
        assert isinstance(backend, SearchBackend)
        assert backend.name == "searxng"

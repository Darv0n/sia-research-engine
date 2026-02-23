"""Tests for config.py — Settings defaults, validation, available_backends."""

from __future__ import annotations

from deep_research_swarm.config import Settings


class TestV7ConfigDefaults:
    """V7 config field defaults (PR-01)."""

    def test_openalex_email_default_empty(self):
        s = Settings(openalex_email="")
        assert s.openalex_email == ""

    def test_openalex_api_key_default_empty(self):
        s = Settings(openalex_api_key="")
        assert s.openalex_api_key == ""

    def test_semantic_scholar_api_key_default_empty(self):
        s = Settings(semantic_scholar_api_key="")
        assert s.semantic_scholar_api_key == ""

    def test_wayback_enabled_default_true(self):
        s = Settings()
        assert s.wayback_enabled is True

    def test_wayback_timeout_default_15(self):
        s = Settings()
        assert s.wayback_timeout == 15


class TestV7AvailableBackends:
    """available_backends() includes V7 scholarly backends."""

    def test_semantic_scholar_always_available(self):
        s = Settings()
        assert "semantic_scholar" in s.available_backends()

    def test_openalex_requires_email(self):
        s = Settings(openalex_email="")
        assert "openalex" not in s.available_backends()

    def test_openalex_available_with_email(self):
        s = Settings(openalex_email="test@example.com")
        assert "openalex" in s.available_backends()

    def test_wayback_available_when_enabled(self):
        s = Settings(wayback_enabled=True)
        assert "wayback" in s.available_backends()

    def test_wayback_not_available_when_disabled(self):
        s = Settings(wayback_enabled=False)
        assert "wayback" not in s.available_backends()

    def test_baseline_backends_unchanged(self):
        """Existing backends still present."""
        s = Settings(exa_api_key="test", tavily_api_key="test")
        backends = s.available_backends()
        assert "searxng" in backends
        assert "exa" in backends
        assert "tavily" in backends

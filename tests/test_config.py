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


class TestV7ConfigWarnings:
    """Config warnings (PR-12)."""

    def test_no_warnings_by_default(self):
        s = Settings()
        assert s.warnings() == []

    def test_aggressive_wayback_timeout_warning(self):
        s = Settings(wayback_enabled=True, wayback_timeout=3)
        warns = s.warnings()
        assert len(warns) == 1
        assert "WAYBACK_TIMEOUT=3s" in warns[0]
        assert "aggressive" in warns[0]

    def test_no_warning_at_threshold(self):
        s = Settings(wayback_enabled=True, wayback_timeout=5)
        assert s.warnings() == []

    def test_no_warning_when_wayback_disabled(self):
        s = Settings(wayback_enabled=False, wayback_timeout=1)
        assert s.warnings() == []


class TestV7BackendRegistryComplete:
    """All V7 backends register when configured (PR-12 integration)."""

    def test_all_v7_backends_available(self):
        s = Settings(
            exa_api_key="test",
            tavily_api_key="test",
            openalex_email="test@test.com",
            wayback_enabled=True,
        )
        available = s.available_backends()
        assert "searxng" in available
        assert "exa" in available
        assert "tavily" in available
        assert "openalex" in available
        assert "semantic_scholar" in available
        assert "wayback" in available
        assert len(available) == 6


class TestV10ConfigDefaults:
    """V10 config field defaults (SIA + Swarm)."""

    def test_sia_enabled_default_true(self):
        s = Settings()
        assert s.sia_enabled is True

    def test_swarm_enabled_default_true(self):
        s = Settings()
        assert s.swarm_enabled is True

    def test_swarm_max_reactors_default_5(self):
        s = Settings()
        assert s.swarm_max_reactors == 5

    def test_sia_can_be_disabled(self):
        s = Settings(sia_enabled=False)
        assert s.sia_enabled is False

    def test_swarm_can_be_disabled(self):
        s = Settings(swarm_enabled=False)
        assert s.swarm_enabled is False

    def test_swarm_max_reactors_custom(self):
        s = Settings(swarm_max_reactors=3)
        assert s.swarm_max_reactors == 3


class TestV10ConfigValidation:
    """V10 swarm_max_reactors validation."""

    def test_swarm_max_reactors_too_low(self):
        s = Settings(swarm_max_reactors=1)
        errors = s.validate()
        assert any("SWARM_MAX_REACTORS" in e for e in errors)

    def test_swarm_max_reactors_too_high(self):
        s = Settings(swarm_max_reactors=11)
        errors = s.validate()
        assert any("SWARM_MAX_REACTORS" in e for e in errors)

    def test_swarm_max_reactors_valid(self):
        s = Settings(swarm_max_reactors=5)
        errors = s.validate()
        assert not any("SWARM_MAX_REACTORS" in e for e in errors)

    def test_swarm_max_reactors_boundary_2(self):
        s = Settings(swarm_max_reactors=2)
        errors = s.validate()
        assert not any("SWARM_MAX_REACTORS" in e for e in errors)

    def test_swarm_max_reactors_boundary_10(self):
        s = Settings(swarm_max_reactors=10)
        errors = s.validate()
        assert not any("SWARM_MAX_REACTORS" in e for e in errors)

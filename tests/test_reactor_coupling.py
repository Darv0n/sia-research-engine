"""Tests for sia/reactor_coupling.py — cross-reactor communication channels."""

from __future__ import annotations

from deep_research_swarm.sia.reactor_coupling import (
    CouplingChannel,
    CouplingMessage,
    CouplingType,
    build_artifact_injection,
    build_entropy_broadcast,
    build_validation_shock,
)


class TestCouplingType:
    """CouplingType enum values."""

    def test_artifact_injection_value(self):
        assert CouplingType.ARTIFACT_INJECTION.value == "artifact_injection"

    def test_entropy_broadcast_value(self):
        assert CouplingType.ENTROPY_BROADCAST.value == "entropy_broadcast"

    def test_validation_shock_value(self):
        assert CouplingType.VALIDATION_SHOCK.value == "validation_shock"


class TestCouplingMessage:
    """CouplingMessage dataclass."""

    def test_create_message(self):
        msg = CouplingMessage(
            source_reactor="r-0",
            target_reactor="r-1",
            coupling_type=CouplingType.ENTROPY_BROADCAST,
            payload={"e": 0.5},
        )
        assert msg.source_reactor == "r-0"
        assert msg.target_reactor == "r-1"
        assert msg.coupling_type == CouplingType.ENTROPY_BROADCAST
        assert msg.payload == {"e": 0.5}
        assert msg.turn == 0

    def test_message_is_frozen(self):
        msg = CouplingMessage(
            source_reactor="r-0",
            target_reactor="r-1",
            coupling_type=CouplingType.ARTIFACT_INJECTION,
            payload={},
        )
        try:
            msg.source_reactor = "changed"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestCouplingChannel:
    """CouplingChannel send/receive/broadcast."""

    def test_send_and_receive(self):
        ch = CouplingChannel()
        msg = CouplingMessage("r-0", "r-1", CouplingType.ENTROPY_BROADCAST, {"e": 0.3})
        ch.send(msg)
        received = ch.receive("r-1")
        assert len(received) == 1
        assert received[0].payload["e"] == 0.3

    def test_receive_filters_by_target(self):
        ch = CouplingChannel()
        ch.send(CouplingMessage("r-0", "r-1", CouplingType.ENTROPY_BROADCAST, {}))
        ch.send(CouplingMessage("r-0", "r-2", CouplingType.ENTROPY_BROADCAST, {}))
        assert len(ch.receive("r-1")) == 1
        assert len(ch.receive("r-2")) == 1
        assert len(ch.receive("r-3")) == 0

    def test_receive_filters_by_type(self):
        ch = CouplingChannel()
        ch.send(CouplingMessage("r-0", "r-1", CouplingType.ENTROPY_BROADCAST, {}))
        ch.send(CouplingMessage("r-0", "r-1", CouplingType.ARTIFACT_INJECTION, {}))
        entropy = ch.receive("r-1", CouplingType.ENTROPY_BROADCAST)
        assert len(entropy) == 1
        artifact = ch.receive("r-1", CouplingType.ARTIFACT_INJECTION)
        assert len(artifact) == 1

    def test_broadcast_skips_self(self):
        ch = CouplingChannel()
        ch.broadcast("r-0", CouplingType.ENTROPY_BROADCAST, {"e": 0.5}, ["r-0", "r-1", "r-2"])
        assert len(ch.receive("r-0")) == 0
        assert len(ch.receive("r-1")) == 1
        assert len(ch.receive("r-2")) == 1

    def test_broadcast_sets_turn(self):
        ch = CouplingChannel()
        ch.broadcast("r-0", CouplingType.ENTROPY_BROADCAST, {}, ["r-1"], turn=3)
        msgs = ch.receive("r-1")
        assert msgs[0].turn == 3

    def test_clear(self):
        ch = CouplingChannel()
        ch.send(CouplingMessage("r-0", "r-1", CouplingType.ENTROPY_BROADCAST, {}))
        ch.clear()
        assert len(ch.messages) == 0


class TestBuildArtifactInjection:
    """build_artifact_injection() helper."""

    def test_creates_messages_for_each_target(self):
        artifact = {
            "clusters": [{"cluster_id": "c1", "theme": "test"}],
            "coverage": {"overall_coverage": 0.7},
            "structural_risks": ["low_authority"],
            "compression_ratio": 0.85,
        }
        msgs = build_artifact_injection("r-0", artifact, ["r-0", "r-1", "r-2"])
        assert len(msgs) == 2  # Skips self
        assert all(m.coupling_type == CouplingType.ARTIFACT_INJECTION for m in msgs)

    def test_payload_contains_lightweight_clusters(self):
        artifact = {
            "clusters": [
                {"cluster_id": "c1", "theme": "topic A", "passages": ["big data..."]},
            ],
            "coverage": {"overall_coverage": 0.8},
            "structural_risks": [],
            "compression_ratio": 0.9,
        }
        msgs = build_artifact_injection("r-0", artifact, ["r-1"])
        payload = msgs[0].payload
        assert "clusters" in payload
        assert payload["clusters"][0]["theme"] == "topic A"
        # Heavy data (passages) should not be in payload
        assert "passages" not in payload["clusters"][0]

    def test_empty_artifact(self):
        msgs = build_artifact_injection("r-0", {}, ["r-1"])
        assert len(msgs) == 1
        assert msgs[0].payload["clusters"] == []


class TestBuildEntropyBroadcast:
    """build_entropy_broadcast() helper."""

    def test_creates_messages(self):
        entropy = {"e": 0.45, "band": "turbulence", "e_amb": 0.4, "e_conf": 0.5}
        msgs = build_entropy_broadcast("r-0", entropy, ["r-1", "r-2"])
        assert len(msgs) == 2
        assert msgs[0].payload["e"] == 0.45
        assert msgs[0].payload["band"] == "turbulence"

    def test_skips_self(self):
        msgs = build_entropy_broadcast("r-0", {"e": 0.3}, ["r-0", "r-1"])
        assert len(msgs) == 1
        assert msgs[0].target_reactor == "r-1"


class TestBuildValidationShock:
    """build_validation_shock() helper."""

    def test_only_propagates_serious_findings(self):
        findings = [
            {"finding": "critical issue", "severity": "critical", "target_section": "sec-1"},
            {"finding": "minor nit", "severity": "minor", "target_section": "sec-2"},
            {"finding": "significant gap", "severity": "significant", "target_section": "global"},
        ]
        msgs = build_validation_shock("r-0", findings, ["r-1"])
        assert len(msgs) == 1
        payload = msgs[0].payload
        assert payload["findings_count"] == 2
        assert len(payload["findings"]) == 2
        severities = {f["severity"] for f in payload["findings"]}
        assert "minor" not in severities

    def test_empty_when_no_serious_findings(self):
        findings = [
            {"finding": "minor nit", "severity": "minor", "target_section": "sec-1"},
        ]
        msgs = build_validation_shock("r-0", findings, ["r-1"])
        assert len(msgs) == 0

    def test_caps_at_10_findings(self):
        findings = [
            {"finding": f"issue {i}", "severity": "critical", "target_section": "global"}
            for i in range(15)
        ]
        msgs = build_validation_shock("r-0", findings, ["r-1"])
        assert len(msgs[0].payload["findings"]) == 10

"""Cross-reactor coupling channels for multi-reactor swarm.

Three coupling types enable information flow between parallel reactors:
  1. artifact_injection — share KnowledgeArtifact clusters between reactors
  2. entropy_broadcast — share entropy state for cross-reactor calibration
  3. validation_shock — inject adversarial findings from one reactor into another

Coupling is opt-in and asynchronous. Each channel produces a CouplingMessage
that the receiving reactor can integrate into its state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CouplingType(Enum):
    """Types of cross-reactor coupling."""

    ARTIFACT_INJECTION = "artifact_injection"
    ENTROPY_BROADCAST = "entropy_broadcast"
    VALIDATION_SHOCK = "validation_shock"


@dataclass(frozen=True)
class CouplingMessage:
    """A message sent between reactors via a coupling channel."""

    source_reactor: str
    target_reactor: str
    coupling_type: CouplingType
    payload: dict[str, Any]
    turn: int = 0


@dataclass
class CouplingChannel:
    """Manages cross-reactor communication for a swarm run.

    Collects messages from completed reactors and dispatches them
    to active reactors based on coupling type.
    """

    messages: list[CouplingMessage] = field(default_factory=list)

    def send(self, message: CouplingMessage) -> None:
        """Queue a coupling message."""
        self.messages.append(message)

    def receive(
        self,
        target_reactor: str,
        coupling_type: CouplingType | None = None,
    ) -> list[CouplingMessage]:
        """Get pending messages for a reactor, optionally filtered by type."""
        return [
            m
            for m in self.messages
            if m.target_reactor == target_reactor
            and (coupling_type is None or m.coupling_type == coupling_type)
        ]

    def broadcast(
        self,
        source_reactor: str,
        coupling_type: CouplingType,
        payload: dict[str, Any],
        target_reactors: list[str],
        turn: int = 0,
    ) -> None:
        """Send the same message to multiple reactors."""
        for target in target_reactors:
            if target != source_reactor:
                self.send(
                    CouplingMessage(
                        source_reactor=source_reactor,
                        target_reactor=target,
                        coupling_type=coupling_type,
                        payload=payload,
                        turn=turn,
                    )
                )

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


def build_artifact_injection(
    source_reactor: str,
    knowledge_artifact: dict[str, Any],
    target_reactors: list[str],
) -> list[CouplingMessage]:
    """Build artifact injection messages from a completed reactor.

    Shares the KnowledgeArtifact's cluster summaries and coverage map
    with other reactors so they can fill gaps the source missed.
    """
    # Extract lightweight payload (not the full artifact)
    payload = {
        "clusters": [
            {"cluster_id": c.get("cluster_id", ""), "theme": c.get("theme", "")}
            for c in knowledge_artifact.get("clusters", [])
        ],
        "coverage": knowledge_artifact.get("coverage", {}),
        "structural_risks": knowledge_artifact.get("structural_risks", []),
        "compression_ratio": knowledge_artifact.get("compression_ratio", 0.0),
    }

    return [
        CouplingMessage(
            source_reactor=source_reactor,
            target_reactor=target,
            coupling_type=CouplingType.ARTIFACT_INJECTION,
            payload=payload,
        )
        for target in target_reactors
        if target != source_reactor
    ]


def build_entropy_broadcast(
    source_reactor: str,
    entropy_state: dict[str, Any],
    target_reactors: list[str],
    turn: int = 0,
) -> list[CouplingMessage]:
    """Build entropy broadcast messages.

    Shares entropy state so other reactors can calibrate their
    convergence behavior relative to the swarm.
    """
    payload = {
        "e": entropy_state.get("e", 0.0),
        "band": entropy_state.get("band", "convergence"),
        "e_amb": entropy_state.get("e_amb", 0.0),
        "e_conf": entropy_state.get("e_conf", 0.0),
    }

    return [
        CouplingMessage(
            source_reactor=source_reactor,
            target_reactor=target,
            coupling_type=CouplingType.ENTROPY_BROADCAST,
            payload=payload,
            turn=turn,
        )
        for target in target_reactors
        if target != source_reactor
    ]


def build_validation_shock(
    source_reactor: str,
    adversarial_findings: list[dict[str, Any]],
    target_reactors: list[str],
) -> list[CouplingMessage]:
    """Build validation shock messages from adversarial critique.

    Injects critical/significant findings from one reactor's critique
    into others, forcing them to address the same issues.
    """
    # Only propagate critical and significant findings
    serious_findings = [
        f for f in adversarial_findings if f.get("severity") in ("critical", "significant")
    ]

    if not serious_findings:
        return []

    payload = {
        "findings": [
            {
                "finding": f.get("finding", ""),
                "severity": f.get("severity", "minor"),
                "target_section": f.get("target_section", "global"),
            }
            for f in serious_findings[:10]  # Cap at 10 to avoid noise
        ],
        "findings_count": len(serious_findings),
    }

    return [
        CouplingMessage(
            source_reactor=source_reactor,
            target_reactor=target,
            coupling_type=CouplingType.VALIDATION_SHOCK,
            payload=payload,
        )
        for target in target_reactors
        if target != source_reactor
    ]

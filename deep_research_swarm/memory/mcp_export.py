"""Export memory records in MCP entity format."""

from __future__ import annotations

import json

from deep_research_swarm.contracts import ResearchMemory


def export_to_mcp_format(memories: list[ResearchMemory]) -> str:
    """Convert memory records to @modelcontextprotocol/server-memory compatible JSON.

    Each memory becomes an entity with type "ResearchRun".
    Observations are key_findings + gaps.
    """
    entities = []
    for mem in memories:
        observations = []
        for finding in mem.get("key_findings", []):
            observations.append(f"Finding: {finding}")
        for gap in mem.get("gaps", []):
            observations.append(f"Gap: {gap}")
        observations.append(
            f"Stats: {mem.get('sources_count', 0)} sources, "
            f"{mem.get('iterations', 0)} iterations, "
            f"converged={mem.get('converged', False)}"
        )

        entities.append(
            {
                "name": mem.get("thread_id", "unknown"),
                "entityType": "ResearchRun",
                "observations": observations,
                "metadata": {
                    "question": mem.get("question", ""),
                    "timestamp": mem.get("timestamp", ""),
                },
            }
        )

    return json.dumps(entities, indent=2, ensure_ascii=False)

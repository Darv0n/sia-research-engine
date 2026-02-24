"""PROV-O JSON-LD export — W3C provenance model (V8, PR-11).

Exports research provenance as PROV-O JSON-LD for interoperability.
ProvenanceRecord maps to prov:Entity, extraction steps to prov:Activity,
and the pipeline to prov:Agent.

See: https://www.w3.org/TR/prov-o/
"""

from __future__ import annotations

import json

from deep_research_swarm.contracts import ProvenanceRecord


def export_prov_o_jsonld(
    provenance_records: list[ProvenanceRecord],
    *,
    research_question: str = "",
    run_id: str = "",
) -> str:
    """Export provenance records as PROV-O JSON-LD string.

    Maps ProvenanceRecord fields to W3C PROV-O terms:
    - Each source -> prov:Entity
    - Each extraction -> prov:Activity
    - The pipeline -> prov:Agent (softwareAgent)

    Returns a JSON-LD string. Empty records produce minimal valid document.
    """
    context = {
        "@context": {
            "prov": "http://www.w3.org/ns/prov#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "drs": "urn:deep-research-swarm:",
        }
    }

    agent = {
        "@id": f"drs:agent/{run_id}" if run_id else "drs:agent/pipeline",
        "@type": "prov:SoftwareAgent",
        "prov:label": "deep-research-swarm",
    }

    entities: list[dict] = []
    activities: list[dict] = []

    for i, prov in enumerate(provenance_records):
        entity_id = prov.get("entity_id", f"drs:entity/{i}")
        activity_id = f"drs:activity/extract-{i}"

        entity = {
            "@id": entity_id,
            "@type": "prov:Entity",
            "prov:value": prov.get("source_url", ""),
            "drs:sourceKind": prov.get("source_kind", "unknown"),
            "drs:contentHash": prov.get("content_hash", ""),
            "drs:licenseTag": prov.get("license_tag", "unknown"),
        }
        entities.append(entity)

        activity = {
            "@id": activity_id,
            "@type": "prov:Activity",
            "prov:used": {"@id": entity_id},
            "prov:wasAssociatedWith": {"@id": agent["@id"]},
            "prov:endedAtTime": prov.get("fetched_at", ""),
            "drs:extractor": prov.get("extractor", ""),
        }

        if prov.get("capture_timestamp"):
            activity["drs:captureTimestamp"] = prov["capture_timestamp"]

        activities.append(activity)

    doc = {
        **context,
        "@graph": [agent, *entities, *activities],
    }

    if research_question:
        doc["drs:researchQuestion"] = research_question

    return json.dumps(doc, indent=2)

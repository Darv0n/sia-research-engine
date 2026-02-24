"""Tests for reporting/prov_o.py — PROV-O JSON-LD export (PR-11)."""

from __future__ import annotations

import json

from deep_research_swarm.contracts import ProvenanceRecord
from deep_research_swarm.reporting.prov_o import export_prov_o_jsonld


def _make_prov(
    url: str = "https://example.com",
    content_hash: str = "abc123",
    extractor: str = "crawl4ai",
    source_kind: str = "web",
) -> ProvenanceRecord:
    return ProvenanceRecord(
        entity_id=f"urn:url:{url}",
        source_url=url,
        source_kind=source_kind,
        fetched_at="2026-02-23T00:00:00Z",
        extractor=extractor,
        license_tag="unknown",
        capture_timestamp="",
        content_hash=content_hash,
    )


class TestExportProvOJsonld:
    def test_produces_valid_json(self):
        provs = [_make_prov()]
        result = export_prov_o_jsonld(provs)
        doc = json.loads(result)
        assert "@context" in doc
        assert "@graph" in doc

    def test_contains_prov_context(self):
        doc = json.loads(export_prov_o_jsonld([_make_prov()]))
        ctx = doc["@context"]
        assert "prov" in ctx
        assert ctx["prov"] == "http://www.w3.org/ns/prov#"

    def test_agent_is_software_agent(self):
        doc = json.loads(export_prov_o_jsonld([_make_prov()]))
        graph = doc["@graph"]
        agent = graph[0]
        assert agent["@type"] == "prov:SoftwareAgent"
        assert "deep-research-swarm" in agent["prov:label"]

    def test_entity_per_provenance(self):
        provs = [_make_prov("https://a.com"), _make_prov("https://b.com")]
        doc = json.loads(export_prov_o_jsonld(provs))
        entities = [n for n in doc["@graph"] if n.get("@type") == "prov:Entity"]
        assert len(entities) == 2

    def test_activity_per_provenance(self):
        provs = [_make_prov()]
        doc = json.loads(export_prov_o_jsonld(provs))
        activities = [n for n in doc["@graph"] if n.get("@type") == "prov:Activity"]
        assert len(activities) == 1
        assert "prov:used" in activities[0]

    def test_entity_has_source_url(self):
        doc = json.loads(export_prov_o_jsonld([_make_prov("https://test.com")]))
        entities = [n for n in doc["@graph"] if n.get("@type") == "prov:Entity"]
        assert entities[0]["prov:value"] == "https://test.com"

    def test_entity_has_content_hash(self):
        doc = json.loads(export_prov_o_jsonld([_make_prov(content_hash="deadbeef")]))
        entities = [n for n in doc["@graph"] if n.get("@type") == "prov:Entity"]
        assert entities[0]["drs:contentHash"] == "deadbeef"

    def test_research_question_included(self):
        doc = json.loads(export_prov_o_jsonld([_make_prov()], research_question="What is quantum?"))
        assert doc["drs:researchQuestion"] == "What is quantum?"

    def test_run_id_in_agent(self):
        doc = json.loads(export_prov_o_jsonld([_make_prov()], run_id="run-123"))
        agent = doc["@graph"][0]
        assert "run-123" in agent["@id"]

    def test_empty_provenance(self):
        result = export_prov_o_jsonld([])
        doc = json.loads(result)
        # Should have just the agent in the graph
        assert len(doc["@graph"]) == 1

    def test_wayback_capture_timestamp(self):
        prov = _make_prov()
        prov["capture_timestamp"] = "20260223120000"
        doc = json.loads(export_prov_o_jsonld([prov]))
        activities = [n for n in doc["@graph"] if n.get("@type") == "prov:Activity"]
        assert activities[0]["drs:captureTimestamp"] == "20260223120000"

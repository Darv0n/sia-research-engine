"""Tests for extractors/grobid_extractor.py — GROBID TEI XML extraction (PR-09)."""

from __future__ import annotations

from deep_research_swarm.extractors.grobid_extractor import _parse_tei_xml

# Minimal TEI XML fixture
SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader/>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>This paper studies quantum entanglement.</p>
        <p>We present novel results on Bell tests.</p>
      </div>
      <div>
        <head>Methods</head>
        <p>We used photon sources and beam splitters.</p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct>
          <analytic>
            <author><persName><surname>Einstein</surname></persName></author>
            <author><persName><surname>Podolsky</surname></persName></author>
            <author><persName><surname>Rosen</surname></persName></author>
            <title>EPR paradox paper</title>
          </analytic>
          <monogr>
            <imprint><date when="1935"/></imprint>
          </monogr>
          <idno type="DOI">10.1103/PhysRev.47.777</idno>
        </biblStruct>
        <biblStruct>
          <analytic>
            <author><persName><surname>Bell</surname></persName></author>
            <title>On the Einstein Podolsky Rosen Paradox</title>
          </analytic>
          <monogr>
            <imprint><date when="1964"/></imprint>
          </monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>"""


class TestParseTeiXml:
    def test_extracts_body_text(self):
        content, _ = _parse_tei_xml(SAMPLE_TEI)
        assert "quantum entanglement" in content
        assert "Bell tests" in content
        assert "photon sources" in content

    def test_extracts_section_headings(self):
        content, _ = _parse_tei_xml(SAMPLE_TEI)
        assert "## Introduction" in content
        assert "## Methods" in content

    def test_extracts_references(self):
        _, refs = _parse_tei_xml(SAMPLE_TEI)
        assert len(refs) == 2
        assert any("Einstein" in r for r in refs)
        assert any("Bell" in r for r in refs)

    def test_reference_has_year(self):
        _, refs = _parse_tei_xml(SAMPLE_TEI)
        epr = next(r for r in refs if "Einstein" in r)
        assert "(1935)" in epr

    def test_reference_has_doi(self):
        _, refs = _parse_tei_xml(SAMPLE_TEI)
        epr = next(r for r in refs if "Einstein" in r)
        assert "doi:10.1103/PhysRev.47.777" in epr

    def test_empty_xml(self):
        content, refs = _parse_tei_xml("")
        assert content == ""
        assert refs == []

    def test_invalid_xml(self):
        content, refs = _parse_tei_xml("<not>valid<xml>")
        assert content == ""
        assert refs == []

    def test_no_body(self):
        xml = '<?xml version="1.0"?><TEI xmlns="http://www.tei-c.org/ns/1.0"><text></text></TEI>'
        content, refs = _parse_tei_xml(xml)
        assert content == ""

    def test_no_references(self):
        xml = """<?xml version="1.0"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text><body><div><p>Some content.</p></div></body></text>
        </TEI>"""
        content, refs = _parse_tei_xml(xml)
        assert "Some content" in content
        assert refs == []

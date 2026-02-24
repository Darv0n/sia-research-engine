"""GROBID extractor — structured PDF extraction via GROBID TEI XML (V8, I8).

GROBID extracts structured sections + reference lists from academic PDFs.
Requires a running GROBID server (Docker or local). Optional dependency.

Returns (content, references) where content is cleaned text with section
headings and references is a list of reference strings for citation chaining.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


def _parse_tei_xml(xml_text: str) -> tuple[str, list[str]]:
    """Parse GROBID TEI XML into clean text + reference list.

    Returns (body_text, references).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return "", []

    # Extract body text with section headings
    body = root.find(f".//{TEI_NS}body")
    sections: list[str] = []

    if body is not None:
        for div in body.iter(f"{TEI_NS}div"):
            head = div.find(f"{TEI_NS}head")
            heading = head.text.strip() if head is not None and head.text else ""
            paragraphs: list[str] = []
            for p in div.findall(f"{TEI_NS}p"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)
            if heading or paragraphs:
                if heading:
                    sections.append(f"## {heading}")
                sections.extend(paragraphs)
                sections.append("")  # blank line between sections

    body_text = "\n".join(sections).strip()

    # Extract references from bibliography
    references: list[str] = []
    for bibl in root.iter(f"{TEI_NS}biblStruct"):
        ref_parts: list[str] = []
        # Authors
        for author in bibl.iter(f"{TEI_NS}author"):
            surname = author.find(f".//{TEI_NS}surname")
            if surname is not None and surname.text:
                ref_parts.append(surname.text)
        # Title
        title = bibl.find(f".//{TEI_NS}title")
        if title is not None and title.text:
            ref_parts.append(title.text)
        # Year
        date = bibl.find(f".//{TEI_NS}date")
        if date is not None:
            year = date.get("when", "")
            if year:
                ref_parts.append(f"({year})")
        # DOI
        for idno in bibl.iter(f"{TEI_NS}idno"):
            if idno.get("type") == "DOI" and idno.text:
                ref_parts.append(f"doi:{idno.text}")

        if ref_parts:
            references.append(" — ".join(ref_parts))

    return body_text, references


async def extract_with_grobid(
    path_or_url: str,
    *,
    grobid_url: str = "",
) -> tuple[str, list[str]]:
    """Extract structured content from PDF via GROBID.

    Args:
        path_or_url: PDF file path or URL.
        grobid_url: GROBID server URL (e.g., "http://localhost:8070").

    Returns (content, references) or ("", []) on failure.
    """
    if not grobid_url:
        return "", []

    # Normalize GROBID URL
    grobid_url = grobid_url.rstrip("/")
    endpoint = f"{grobid_url}/api/processFulltextDocument"

    pdf_bytes: bytes | None = None
    tmp_path: str | None = None

    try:
        if path_or_url.startswith(("http://", "https://")):
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(path_or_url)
                resp.raise_for_status()
                pdf_bytes = resp.content
        else:
            pdf_bytes = await asyncio.to_thread(Path(path_or_url).read_bytes)

        if not pdf_bytes:
            return "", []

        # Write to temp file for multipart upload
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        Path(tmp_path).write_bytes(pdf_bytes)

        # Send to GROBID
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(tmp_path, "rb") as f:
                resp = await client.post(
                    endpoint,
                    files={"input": ("document.pdf", f, "application/pdf")},
                )
                resp.raise_for_status()

        tei_xml = resp.text
        if not tei_xml or len(tei_xml) < 100:
            return "", []

        return _parse_tei_xml(tei_xml)

    except Exception:
        return "", []
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()

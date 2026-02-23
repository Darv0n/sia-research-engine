"""Passage chunking for extracted documents.

Splits extracted content into SourcePassage objects with deterministic,
content-hash-based IDs (D1, I1). 4-tier chunking strategy:
1. Markdown headings
2. Double newlines (paragraph boundaries)
3. Sentence boundaries
4. Hard split at 2x target_tokens

Protects markdown tables and code blocks from mid-split.
"""

from __future__ import annotations

import hashlib
import re

from deep_research_swarm.contracts import ExtractedContent, ScoredDocument, SourcePassage


def _make_passage_id(source_id: str, position: int) -> str:
    """Deterministic passage ID from source_id + position (D1, I1).

    Same input -> same IDs. Required for cross-run dedup,
    incremental research, and claim graph traversal.
    """
    raw = (source_id + str(position)).encode()
    return f"sp-{hashlib.sha256(raw).hexdigest()[:8]}"


def _estimate_tokens(text: str) -> int:
    """Estimate token count as word_count * 1.3."""
    words = len(text.split())
    return max(1, int(words * 1.3))


def _split_respecting_blocks(text: str, pattern: str) -> list[str]:
    """Split text by pattern but protect code blocks and markdown tables.

    Code blocks (``` fences) and table rows (lines starting with |)
    are never split mid-block.
    """
    # Identify protected regions (code blocks)
    code_block_re = re.compile(r"```.*?```", re.DOTALL)
    protected: list[tuple[int, int]] = []
    for m in code_block_re.finditer(text):
        protected.append((m.start(), m.end()))

    # Also protect contiguous table rows
    table_row_re = re.compile(r"((?:^\|.*$\n?)+)", re.MULTILINE)
    for m in table_row_re.finditer(text):
        protected.append((m.start(), m.end()))

    def _in_protected(pos: int) -> bool:
        return any(start <= pos < end for start, end in protected)

    # Find all split positions
    chunks: list[str] = []
    last = 0
    for m in re.finditer(pattern, text):
        if _in_protected(m.start()):
            continue
        chunk = text[last : m.start()].strip()
        if chunk:
            chunks.append(chunk)
        last = m.end()

    # Remainder
    tail = text[last:].strip()
    if tail:
        chunks.append(tail)

    return chunks if chunks else [text.strip()] if text.strip() else []


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split by markdown headings, returning (heading_context, chunk) pairs."""
    heading_re = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)
    parts: list[tuple[str, str]] = []
    current_heading = ""
    last_pos = 0

    for m in heading_re.finditer(text):
        before = text[last_pos : m.start()].strip()
        if before:
            parts.append((current_heading, before))
        current_heading = m.group(1).strip()
        last_pos = m.end()

    tail = text[last_pos:].strip()
    if tail:
        parts.append((current_heading, tail))

    return parts if parts else [("", text.strip())] if text.strip() else []


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries."""
    # Split on period/question/exclamation followed by space+uppercase
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, max_words: int) -> list[str]:
    """Hard split at word boundary when no natural boundary found."""
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def _chunk_text(
    text: str,
    heading_context: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str]]:
    """Chunk a single text block into (heading, content) pairs.

    Uses paragraph -> sentence -> hard split cascade.
    Returns chunks within target size.
    """
    target_words = int(target_tokens / 1.3)
    max_words = target_words * 2  # Hard limit at 2x

    # If short enough, return as-is
    if _estimate_tokens(text) <= target_tokens:
        return [(heading_context, text)]

    # Try paragraph split first
    paragraphs = _split_respecting_blocks(text, r"\n\n+")

    result_chunks: list[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) <= target_tokens:
            result_chunks.append(para)
        elif _estimate_tokens(para) <= target_tokens * 2:
            # Try sentence split
            sentences = _split_sentences(para)
            current: list[str] = []
            current_len = 0
            for sent in sentences:
                sent_tokens = _estimate_tokens(sent)
                if current_len + sent_tokens > target_tokens and current:
                    result_chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += sent_tokens
            if current:
                result_chunks.append(" ".join(current))
        else:
            # Hard split
            result_chunks.extend(_hard_split(para, max_words))

    # Apply overlap: prepend last overlap_tokens words of chunk K to chunk K+1
    if overlap_tokens > 0 and len(result_chunks) > 1:
        overlap_words = max(1, int(overlap_tokens / 1.3))
        overlapped: list[str] = [result_chunks[0]]
        for i in range(1, len(result_chunks)):
            prev_words = result_chunks[i - 1].split()
            overlap_prefix = " ".join(prev_words[-overlap_words:])
            overlapped.append(overlap_prefix + " " + result_chunks[i])
        result_chunks = overlapped

    return [(heading_context, chunk) for chunk in result_chunks]


def chunk_document(
    extracted_content: ExtractedContent,
    scored_doc: ScoredDocument,
    *,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[SourcePassage]:
    """Chunk an extracted document into SourcePassage objects.

    4-tier strategy: headings -> paragraphs -> sentences -> hard split.
    IDs are deterministic and content-hash-based (D1, I1).
    """
    content = extracted_content.get("content", "")
    if not content or not content.strip():
        return []

    url = scored_doc["url"]
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    source_id = url + content_hash

    # Stage 1: Split by headings
    heading_sections = _split_by_headings(content)

    # Stage 2: Chunk each section
    all_chunks: list[tuple[str, str]] = []
    for heading, section_text in heading_sections:
        chunks = _chunk_text(
            section_text,
            heading,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(chunks)

    # Stage 3: Build SourcePassage objects
    passages: list[SourcePassage] = []
    char_offset = 0

    for position, (heading, chunk_content) in enumerate(all_chunks):
        # Find actual char offset in original content
        idx = content.find(chunk_content[:50], char_offset)
        if idx >= 0:
            char_offset = idx

        passage = SourcePassage(
            id=_make_passage_id(source_id, position),
            source_id=source_id,
            source_url=url,
            content=chunk_content,
            position=position,
            char_offset=max(0, char_offset),
            token_count=_estimate_tokens(chunk_content),
            heading_context=heading,
            claim_ids=[],  # OE1: empty in V7, populated in V8
        )
        passages.append(passage)

    return passages


def chunk_all_documents(
    extracted_contents: list[ExtractedContent],
    scored_documents: list[ScoredDocument],
    *,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[SourcePassage]:
    """Chunk all documents, matching by URL. Skip unmatched silently."""
    scored_by_url: dict[str, ScoredDocument] = {}
    for sd in scored_documents:
        scored_by_url[sd["url"]] = sd

    all_passages: list[SourcePassage] = []
    for ec in extracted_contents:
        if not ec.get("extraction_success", False):
            continue
        url = ec["url"]
        scored = scored_by_url.get(url)
        if scored is None:
            continue
        passages = chunk_document(
            ec,
            scored,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_passages.extend(passages)

    return all_passages

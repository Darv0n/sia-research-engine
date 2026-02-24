# V10: Knowledge Compression Layer — Research-Backed Plan

## Context: The Bottleneck, Precisely

From the V9 astrology run stderr logs:
- **Total cost: $1.82** — 377K tokens across 3 iterations
- **Synthesis cost: $1.55 (85%)** — 301K tokens, 19 minutes of Opus
- **5-stage pipeline**: outline → draft → verify → refine → compose = 14-19 LLM calls/iteration
- **All on Opus** ($15/$75 per 1M tokens) — the most expensive model

The synthesizer (`synthesizer.py:157-172`) calls `_build_source_context()` which takes top 20
scored docs, sends 1000 chars each as flat text to Opus. It ignores: authority scores,
claim_ids (V8, populated but unused), passage grounding data, diversity metrics, provenance.
Then it runs 3 more Opus passes to verify and fix what it wrote.

**19,056 passages produced → ~160 consumed = 99.2% waste ratio.**

The pipeline upstream computes rich structure. The synthesizer throws it away and asks Opus
to rediscover it from scratch. Five times.

## The Research: What Actually Works

### Tier 1 — Techniques with verified numbers (papers + benchmarks)

| Technique | Paper | Core Idea | Token Reduction | Model/Cost |
|-----------|-------|-----------|-----------------|------------|
| **CompactRAG** | arXiv:2602.05728 (Feb 2026) | Offline atomic QA decomposition → 2 LLM calls per query | **81% vs IRCoT** (1.9K vs 10.2K tokens/sample) | RoBERTa-base extraction (free) + Flan-T5-small rewrite |
| **CASC** | arXiv:2508.19357 (Aug 2025) | Fine-tuned 7B CAS module between retrieval and generation | **68% compression** (1280→405 tokens), hallucination 18.2%→6.1% | Llama-2-7B (heavy dep) |
| **MiniCheck** | arXiv:2404.10774 (EMNLP 2024) | 770M fact-checker, GPT-4 accuracy on grounding | Eliminates LLM verify stage entirely | `lytang/MiniCheck-Flan-T5-Large` (MIT, 64.6K downloads) — **$0.24/13K claims vs $107 GPT-4** |
| **NLI Cross-Encoder** | sentence-transformers | 3-way entailment/contradiction/neutral | Replaces LLM contradiction detection | `cross-encoder/nli-deberta-v3-xsmall` (Apache-2.0, 70.8M params, ONNX, **CPU-viable**) |
| **RAPTOR** | arXiv:2401.18059 (ICLR 2024) | UMAP+GMM soft clustering → recursive summarization tree | **72% compression per tree level** (131-token summary from avg 86-token children) | Clustering is LLM-free; summaries use cheap model |
| **LLMLingua-2** | arXiv:2403.12968 (Microsoft) | Token-level compression via XLM-RoBERTa classification | **67% token removal** at 33% retention rate | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (MIT, 559M, **CPU OK**) |
| **CRAG** | arXiv:2401.15884 (2024) | Confidence-gated retrieval: correct/ambiguous/incorrect routing | Prevents synthesis on garbage input | T5-large evaluator (plug-and-play wrapper) |
| **ODR Pattern** | langchain-ai/open_deep_research | Cheap model compresses sub-agent findings before expensive synthesis | **~50% context reduction** | Uses gpt-4.1-mini; we'd use **Haiku** ($0.25/$1.25 per 1M) |
| **StructRAG** | arXiv:2410.08815 (2024) | Route query → optimal structure (table/graph/catalogue/chunk) | **22x faster than GraphRAG** | DPO-trained router (pattern, not dep) |
| **CORE-RAG** | arXiv:2508.19282 (Aug 2025) | RL-trained compression policy (GRPO) | **97% token removal**, +3.3 EM improvement | Training approach (pattern) |

### Tier 2 — Models verified on HuggingFace (via MCP)

| Model ID | Params | License | Downloads | GPU? | Our Use |
|----------|--------|---------|-----------|------|---------|
| `BAAI/bge-small-en-v1.5` | 33.4M | MIT | 106.5M | No (ONNX) | **ALREADY IN DEPS** (fastembed) — passage clustering |
| `cross-encoder/nli-deberta-v3-xsmall` | 70.8M | Apache-2.0 | 429.8K | No (ONNX) | NLI grounding + contradiction detection |
| `cross-encoder/nli-deberta-v3-base` | 184.4M | Apache-2.0 | 3.7M | No (ONNX) | Higher-accuracy NLI (optional upgrade) |
| `lytang/MiniCheck-Flan-T5-Large` | 770M | MIT | 64.6K | Slow CPU | Dedicated fact-checker (optional, heavy) |
| `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | 559M | MIT | 1.8M | No | Token compression (optional, heavy) |

## The Math: Current V9 Cost Breakdown

### Per-iteration token accounting (from codebase analysis)

| Stage | Calls | Model | Input | Output | Cost/iter |
|-------|-------|-------|-------|--------|-----------|
| Planner | 1 | Opus | 600-2,400 | 1,500 | $0.12-0.23 |
| Synthesizer outline | 1-2 | Opus | 2,300 | 1,200 | $0.12-0.24 |
| Synthesizer drafts | 4-8 | Opus | 3,200-6,400 | 1,600-3,200 | $0.17-0.34 |
| Synthesizer refine | 2-8 | Opus | 2,200-8,800 | 800-3,200 | $0.09-0.34 |
| Synthesizer compose | 1 | Opus | 500 | 550 | $0.05 |
| Critic (3 graders) | 3 | Sonnet | **9,300** (3x dupe) | 450 | $0.03 |
| Contradiction | 1 | Sonnet | 2,500 | 400 | $0.01 |
| **Subtotal** | 13-24 | — | 20,600-33,200 | 6,500-10,500 | **$0.59-1.24** |

Key waste: critic sends identical `sections_text` to 3 separate Sonnet calls = 6,200 tokens
duplicated per iteration. Synthesizer refine stage exists only because outline+draft input
wasn't structured enough → refinement is compensating for poor input.

### 3-iteration total (typical): ~$1.82

**Cost by model tier:**
- Opus (planner + synthesizer): ~$1.55 (85%)
- Sonnet (critic + contradiction): ~$0.27 (15%)

**Cost by stage:**
- Synthesis (outline+draft+verify+refine+compose): ~$1.12 (62%)
- Replanning (iter 2-3 planner): ~$0.35 (19%)
- Critique (3-grader chain x3): ~$0.09 (5%)
- Other (contradiction, gap analysis): ~$0.26 (14%)

## V10 Target: The Compression Architecture

```
V9:  scored_passages ──────────────────────────> Opus (5 stages, $1.55)
     19K passages → flat top-20 → outline → draft → verify → refine → compose

V10: scored_passages → COMPRESS → artifact → Sonnet (2 stages, $0.15-0.25)
     19K passages → cluster → rank → verify claims → summarize → outline → draft
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    Deterministic + Haiku (~$0.01)    Sonnet (not Opus)
```

### V10 per-iteration cost projection

| Stage | Calls | Model | Input | Output | Cost/iter |
|-------|-------|-------|-------|--------|-----------|
| Planner | 1 | Opus | 600-2,400 | 1,500 | $0.12-0.23 |
| **Compress (cluster+rank)** | 0 | N/A | 0 | 0 | **$0.00** |
| **Compress (Haiku summaries)** | 8-12 | **Haiku** | 6,000-9,000 | 1,600-2,400 | **$0.005** |
| **Compress (NLI grounding)** | 0 | Local model | 0 | 0 | **$0.00** |
| Outline from artifact | 1 | Opus | **2,000** (compact) | 1,200 | $0.12 |
| Section drafts | 8 | **Sonnet** | 4,000 | 3,200 | **$0.06** |
| Composition | 1 | Haiku | 500 | 550 | $0.001 |
| Critic (single call) | 1 | Sonnet | **3,100** (no dupe) | 450 | **$0.01** |
| Contradiction (NLI) | 0 | Local model | 0 | 0 | **$0.00** |
| **Subtotal** | 20-23 | — | 16,200-22,150 | 8,500-9,300 | **$0.32-0.42** |

### 3-iteration projection: $0.95-1.25 (vs $1.82 = 31-48% reduction)

But with adaptive convergence (coverage-gated stopping):
- Simple queries: 1 iteration → **$0.32-0.42**
- Moderate queries: 2 iterations → **$0.65-0.85**
- Complex queries: 3 iterations → **$0.95-1.25**

**Weighted average (est. 40/40/20 distribution): $0.55-0.75 = 59-70% reduction**

### Where the savings come from (exact mechanisms)

| Mechanism | Tokens Saved/iter | Cost Saved/iter | Source |
|-----------|-------------------|-----------------|--------|
| Sonnet for sections (not Opus) | 0 tokens | $0.19 | Opus output=$0.075/Ktok → Sonnet=$0.015/Ktok |
| Drop verify+refine stages | 6,000-17,600 | $0.09-0.34 | Grounding done in compress (NLI, free) |
| Compact artifact input to outline | 1,500-4,500 | $0.02-0.07 | 2K input (artifact JSON) vs 5K (flat passages) |
| Single critic call (not 3) | 6,200 | $0.02 | Eliminate 2x duplication of sections_text |
| NLI contradiction (not Sonnet) | 2,900 | $0.01 | Local model replaces Sonnet call |
| Haiku cluster summaries (not Opus) | 0 tokens | $0.10 | Haiku=$0.25/MTok vs Opus=$15/MTok (60x cheaper) |
| **Total per iteration** | **16,600-30,700** | **$0.43-0.73** | |

## Dependency Strategy

### Tier 0: Zero new dependencies (ship immediately)
- **Fastembed clustering** — `BAAI/bge-small-en-v1.5` already in `[embeddings]` optional dep
- **Haiku compression** — Anthropic SDK already installed, just add `claude-haiku-4-5-20251001`
- **Deterministic artifact builder** — pure Python, uses existing scored_documents + source_passages
- **Single-call critic** — prompt redesign, no new code

### Tier 1: Light optional dep (~35 MB)
- **sklearn** — GMM soft clustering (RAPTOR pattern), agglomerative clustering with cosine
  - `pip install scikit-learn` — well-maintained, BSD license
  - Enables: optimal cluster count via BIC, soft assignment (passages in multiple clusters)

### Tier 2: Heavy optional dep (~2 GB, requires torch+transformers)
- **sentence-transformers** — NLI cross-encoder for grounding + contradiction
  - `cross-encoder/nli-deberta-v3-xsmall` (70.8M) or `-base` (184M)
  - `pip install "sentence-transformers[onnx]"` for CPU inference
  - Replaces Jaccard entirely: 3-way entailment/contradiction/neutral
- **MiniCheck** — dedicated fact-checker (alternative to NLI cross-encoder)
  - `pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"`
  - 770M params, MIT license
  - Only worth it if cross-encoder accuracy insufficient

### Decision: Ship Tier 0 first, Tier 1 fast-follow, Tier 2 optional

Tier 0 alone delivers 40-50% cost reduction. Tier 1 improves clustering quality.
Tier 2 upgrades grounding accuracy from "crude Jaccard" to "GPT-4 level NLI" but
adds 2 GB to install size. Make it an optional dep group like `[nli]`.

## Implementation Plan

### Phase 1: Compression Layer (Tier 0 — zero new deps)

**New module: `deep_research_swarm/compress/`**

**`compress/__init__.py`** — module marker

**`compress/cluster.py`** — passage clustering
- `cluster_by_embedding(passages, provider, k=None)` — k-means via fastembed
  - Uses existing `FastEmbedProvider` from `scoring/embedding_grounding.py`
  - Farthest-first centroid init (deterministic, no random seed)
  - `estimate_k(n_passages)` — heuristic: `clamp(sqrt(n/2), 3, 12)`
  - Falls back to `cluster_by_heading(passages)` when embeddings unavailable
- `rank_passages_in_cluster(passages, scored_documents, max_passages)` — reuse combined_score
- `label_cluster_theme(passages)` — most common heading_context

**`compress/artifact.py`** — knowledge artifact builder
- `build_knowledge_artifact(question, scored_docs, passages, contradictions, ...)`
  1. Cluster passages (embedding or heading fallback)
  2. Rank within clusters by combined_score from scored_documents
  3. Extract sentence-level claims from top passages
  4. Verify claims via cascade: NLI → embedding → Jaccard (graceful degradation)
  5. Build authority profile per cluster from scored_documents
  6. Compute coverage map: decompose question → match facets vs cluster themes
  7. Detect cross-cluster tensions (negation pattern heuristic)
  8. Return `KnowledgeArtifact` TypedDict

**`compress/grounding.py`** — claim verification cascade
- `verify_claim_grounding(claim, passage, *, nli_model=None, embedding_provider=None)`
  - Tier 1: NLI cross-encoder (if available) → entailment/contradiction/neutral
  - Tier 2: Embedding cosine similarity (if fastembed available)
  - Tier 3: Jaccard (always available)
- `NLIVerifier` class — lazy-loads cross-encoder, CPU/ONNX
  - Protocol-compatible with existing `EmbeddingProvider` pattern

**`contracts.py`** additions:
- `ClusterClaim`, `AuthorityProfile`, `ClusterPassage` TypedDicts
- `KnowledgeCluster`, `CrossClusterTension`, `CoverageMap` TypedDicts
- `KnowledgeArtifact` TypedDict (the artifact itself)

**`graph/state.py`** addition:
- `knowledge_artifact: Annotated[KnowledgeArtifact, _replace_dict]`

**`graph/builder.py`** changes:
- Import + instantiate compression components
- New `compress_node(state, config)` — calls `build_knowledge_artifact()`
- Stream `compression_summary` event (clusters, claims, coverage, ratio)
- Wire: `gap_analysis → [follow-up?] → compress → adapt_synthesis`
  (compress sits between scoring and synthesis, reads scored data, writes artifact)

### Phase 2: 2-Stage Synthesis (the cost cut)

**Rewrite `agents/synthesizer.py`:**

Current 5-stage → New 2-stage:

**Stage 1: Outline from Artifact (1 Opus call)**
- Input: KnowledgeArtifact JSON (compact — themes, claims, authority, gaps, tensions)
  - NOT flat passages. The LLM sees pre-structured knowledge.
  - Estimated input: ~2,000 tokens (vs ~5,000 current)
- Prompt redesign: "Given this knowledge structure, assign clusters to sections"
- Output: section headings + assigned cluster indices + key claims per section

**Stage 2: Parallel Section Drafts (N Sonnet calls)**
- Input per section: cluster summary + top 5 pre-ranked passages + pre-verified claims
  - Each passage already has authority_score and grounding status
  - Estimated input: ~500 tokens per section (vs ~800 current)
- **Sonnet, not Opus** — structured input means cheaper model is sufficient
- No verify/refine stages — claims pre-grounded in compress layer

**Composition: 1 Haiku call** (or deterministic template)
- Intro/transitions/conclusion from outline structure
- Haiku at $0.25/$1.25 per 1M tokens — effectively free

**Drop from synthesizer:**
- Stage 3 (`compute_section_grounding_score` loop) — moved to compress
- Stage 4 (`_refine_section` loop) — eliminated by pre-grounded input
- Stage 5 (`_compose_report` Opus call) — replaced by Haiku

**Backward compatibility:**
- Output unchanged: `section_drafts`, `citations`, `citation_to_passage_map`, `token_usage`
- Critic still receives `section_drafts` with same shape
- Renderer still receives same state fields

### Phase 3: Critic Consolidation + Convergence

**Single-call critic** (replace 3-grader chain):
- One prompt: "Evaluate on relevance, hallucination, quality — output all three scores per section"
- Saves 6,200 tokens/iteration (2x duplication eliminated)
- Same output shape: `GraderScores` per section

**NLI contradiction detection** (replace Sonnet contradiction_node):
- If `sentence-transformers` available: cross-encoder NLI on claim pairs
- Cross-cluster tension detection from artifact (already computed in Phase 1)
- Falls back to existing Sonnet call when NLI unavailable
- Saves 2,900 tokens/iteration when NLI available

**Coverage-gated convergence** (replace static max_iterations):
- After compress: check `coverage_map.coverage_score`
- If coverage ≥ 0.8 AND average claim confidence ≥ 0.7 → converge early
- If coverage < 0.5 → targeted re-search (only missing facets)
- Deterministic, zero tokens — just reads artifact metadata

### Phase 4: Tests + CLAUDE.md + Release Artifacts

- Tests for: cluster.py, artifact.py, grounding.py, rewritten synthesizer, single-call critic
- Target: 850+ tests (777 existing + ~75 new)
- Update CLAUDE.md: V10 architecture, new nodes, new state fields, new tunables
- Update pyproject.toml: version 0.10.0, new optional deps `[nli]`, `[clustering]`
- Ruff check + format

## Verification

After each phase:
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
```

After Phase 2 (the money shot):
```bash
# Same astrology query, compare cost/time/quality
.venv/Scripts/python.exe -m deep_research_swarm \
  "astrology sources for professional astrologers" --verbose
# Target: <$0.80, <15 minutes, comparable or better section quality
```

## Critical Files

| File | Change | Phase |
|------|--------|-------|
| `contracts.py` | Add 7 TypedDicts (KnowledgeArtifact family) | 1 |
| `graph/state.py` | Add `knowledge_artifact` field | 1 |
| `graph/builder.py` | Add compress_node, wire edges, init providers | 1 |
| `agents/synthesizer.py` | **Rewrite**: 5-stage → 2-stage from artifact | 2 |
| `agents/critic.py` | Single-call 3-dimension grader | 3 |
| `agents/contradiction.py` | NLI cascade (optional dep) | 3 |
| `adaptive/registry.py` | Add compression tunables | 1 |
| `config.py` | Add `HAIKU_MODEL` env var | 2 |
| `pyproject.toml` | Version bump, new optional dep groups | 4 |
| NEW: `compress/__init__.py` | Module marker | 1 |
| NEW: `compress/cluster.py` | Passage clustering (fastembed/heading) | 1 |
| NEW: `compress/artifact.py` | KnowledgeArtifact builder | 1 |
| NEW: `compress/grounding.py` | NLI/embedding/Jaccard cascade | 1 |

## Research Sources

- CompactRAG: [arXiv:2602.05728](https://arxiv.org/abs/2602.05728) — 2 LLM calls, 81% token reduction
- CASC: [arXiv:2508.19357](https://arxiv.org/abs/2508.19357) — 68% compression, 6.1% hallucination
- MiniCheck: [arXiv:2404.10774](https://arxiv.org/abs/2404.10774) — 400x cheaper than GPT-4 verification
- RAPTOR: [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) — 72% compression per tree level
- StructRAG: [arXiv:2410.08815](https://arxiv.org/abs/2410.08815) — 22x faster than GraphRAG
- CRAG: [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) — confidence-gated retrieval
- Self-RAG: [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) — adaptive retrieval tokens
- GraphRAG: [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) — community detection + hierarchy
- LLMLingua-2: [arXiv:2403.12968](https://arxiv.org/abs/2403.12968) — 67% token compression, CPU
- CORE-RAG: [arXiv:2508.19282](https://arxiv.org/abs/2508.19282) — 97% compression via RL
- ODR: [github.com/langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) — Haiku compression pattern
- STORM: [github.com/stanford-oval/storm](https://github.com/stanford-oval/storm) — perspective discovery
- HuggingFace MCP: models verified via `hub_repo_details` — params, license, ONNX availability confirmed

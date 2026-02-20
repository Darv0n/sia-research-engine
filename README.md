# Deep Research Swarm

Multi-agent deep research system that decomposes questions via STORM-style perspective-guided questioning, dispatches parallel search agents, synthesizes via RAG-Fusion with Reciprocal Rank Fusion, critiques via grader chain, and iterates until convergence.

## Quick Start

```bash
# 1. Start SearXNG
docker compose -f docker/docker-compose.yml up -d

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 4. Run
python -m deep_research_swarm "What is quantum entanglement?"
```

## Architecture

```
START -> plan -> search(parallel) -> extract(parallel)
      -> score -> synthesize -> critique
      -> [converged?] -- YES -> report -> END
                      -- NO  -> plan (loop)
```

- **Planner**: STORM-style decomposition into sub-queries with perspectives
- **Searcher**: Parallel search across backends (SearXNG, Exa, Tavily)
- **Extractor**: Cascade extraction (Crawl4AI -> Trafilatura -> PyMuPDF4LLM)
- **Scorer**: Reciprocal Rank Fusion + source authority
- **Synthesizer**: RAG-Fusion synthesis with inline citations
- **Critic**: Quality grading with convergence detection

## Testing

```bash
pytest tests/ -k "not integration" -v
```

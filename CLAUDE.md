# Deep Research Swarm

## Architecture

LangGraph StateGraph orchestrating multi-agent research pipeline:
`plan -> search(parallel) -> extract(parallel) -> score -> synthesize -> critique -> [converge?]`

## Key Files

- `contracts.py` — Single source of truth for all types, enums, protocols
- `config.py` — All settings from env vars
- `graph/state.py` — ResearchState with annotated reducers
- `graph/builder.py` — Graph construction and wiring

## Conventions

- All agent nodes: `async def node_name(state: ResearchState) -> dict`
- List fields use `Annotated[list[T], operator.add]` for concurrent merge
- Backends implement `SearchBackend` Protocol (structural subtyping)
- Opus for planning/synthesis/critique, Sonnet for search/extraction

## Testing

```bash
pytest tests/ -k "not integration" -v   # Unit tests (no network)
pytest tests/test_integration.py -v      # Mocked LLM, real graph
```

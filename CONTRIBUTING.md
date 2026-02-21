# Contributing

## Dev Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"  # Windows
# .venv/bin/pip install -e ".[dev]"    # Linux/macOS
cp .env.example .env
# Fill in ANTHROPIC_API_KEY at minimum
```

## Running Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v                    # All tests
.venv/Scripts/python.exe -m pytest tests/ -k "not integration"  # Unit only
.venv/Scripts/python.exe -m pytest tests/test_graph.py -v       # Single file
```

## Linting

```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m ruff format .   # Auto-fix formatting
```

## Architecture Conventions

### Single Source of Truth

All TypedDicts, Enums, and Protocols live in `contracts.py`. Import from there, never redefine types elsewhere.

### Node Closure Pattern

Graph nodes are defined as closures inside `build_graph()` in `graph/builder.py`. Each closure captures `settings`, agent callers, and other dependencies from the enclosing scope.

```python
async def my_node(state: ResearchState) -> dict:
    # Access settings, opus_caller, sonnet_caller via closure
    return {"field": value}
```

### Reducer Discipline

- **Accumulating fields** use `Annotated[list[T], operator.add]` — values append across iterations
- **Current-view fields** use `Annotated[T, _replace_*]` — last write wins

Never mix these. If a field accumulates, it always accumulates.

### Model Tiering

- **Opus**: Planning and synthesis (complex reasoning)
- **Sonnet**: Critique, contradiction detection (fast structured analysis)

### Fan-out Nodes

Nodes that do parallel work (search, extract) accept a second parameter for stream writer access:

```python
async def fan_out_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
    writer = _get_stream_writer(config)
    # ...
```

## Adding a New Node

1. Define any new types in `contracts.py`
2. Add any new config vars to `config.py` Settings with env var defaults
3. Add state fields to `graph/state.py` with appropriate reducers
4. Write the node function as a closure in `build_graph()` (`graph/builder.py`)
5. Wire edges: `graph.add_node(...)` and `graph.add_edge(...)`
6. Add a display label to `NODE_LABELS` in `streaming.py`
7. Write tests in `tests/test_<feature>.py` using existing fixture patterns
8. Update `.env.example` and `CLAUDE.md` if new config vars were added

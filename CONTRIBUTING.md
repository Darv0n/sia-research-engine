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

## Release Workflow

### Pre-work (before starting a new version)

1. **Verify baseline** — tests pass, lint clean, working tree clean
   ```bash
   .venv/Scripts/python.exe -m pytest tests/ -v
   .venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
   git status
   ```
2. **Read current state** — `CLAUDE.md` (version, test count, deferred items), `CHANGELOG.md`
3. **Plan scope** — define features, estimate commits, identify files to touch
4. **Create task list** — track progress through the implementation

### During work

- Commit in logical units (type first, then module, then tests, then wiring)
- Run tests after each feature, not just at the end
- Run lint before committing — `ruff format` auto-fixes most issues
- Use conventional commit prefixes: `feat`, `fix`, `test`, `docs`, `chore`
- Live test when the full pipeline is wired up

### Post-work (after completing a version)

1. **Final verification** — full test suite + lint
2. **Update CLAUDE.md** — version, test count, architecture diagram, new files, CLI flags, config vars, deferred items
3. **Update CHANGELOG.md** — new version section with Added/Changed/Fixed
4. **Update .env.example** — if new config vars were added
5. **Commit docs** — single `docs(vN)` commit for all doc updates
6. **Tag** — `git tag vX.Y.Z`
7. **Push** — `git push origin main --tags`
8. **PR** — create with summary, stats, live test results, test plan checklist
9. **Merge + clean up** — merge PR, delete feature branch (local + remote)
10. **Update session memory** — version, test count, what shipped, deferred items

### Artifacts to clean up

- `runs/` — event log JSONL from test runs (gitignored)
- `output/` — generated reports from test runs (gitignored except .gitkeep)
- `stderr.log` — streaming output captured during test runs (not tracked)
- Stale checkpoint DBs in `checkpoints/` (gitignored)

## Adding a New Node

1. Define any new types in `contracts.py`
2. Add any new config vars to `config.py` Settings with env var defaults
3. Add state fields to `graph/state.py` with appropriate reducers
4. Write the node function as a closure in `build_graph()` (`graph/builder.py`)
5. Wire edges: `graph.add_node(...)` and `graph.add_edge(...)`
6. Add a display label to `NODE_LABELS` in `streaming.py`
7. Write tests in `tests/test_<feature>.py` using existing fixture patterns
8. Update `.env.example` and `CLAUDE.md` if new config vars were added

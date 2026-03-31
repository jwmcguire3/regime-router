# AGENTS.md

## Project

Cognitive Router — a Python 3.10+ runtime for routing LLM tasks across six reasoning regimes (exploration, synthesis, epistemic, adversarial, operator, builder). Behavioral control is external to the controlled system: the router selects a regime, composes instruction lines from a grammar, and validates the LLM output against stage-specific contracts.

## Repository layout

```
router/                     # Main package
  models.py                 # All data models, LinePrimitive LIBRARY, Stage enum
  analyzer.py               # LLM-based task analyzer (primary routing proposer)
  classifier.py             # Direct-execution bypass classifier
  routing.py                # Router class, module-level re-exports
  routing/                  # Routing subsystem
    feature_extraction.py   # Structural signal extraction from task text
    risk_inference.py       # Risk profile inference
    composer.py             # RegimeComposer (delegates to GrammarComposer)
    grammar_composer.py     # Grammar-driven regime composition
    grammar_rules.py        # Composition grammar validation
    failure_selection.py    # Failure-cost-driven line selection
    lexical_tables.py       # DEPRECATED — lexical phrase tables
    score_tracking.py       # DEPRECATED — stage score bookkeeping
    confidence.py           # DEPRECATED — confidence calculation
    decision_builder.py     # DEPRECATED — score-based decision building
    analyzer_override.py    # DEPRECATED — analyzer acceptance gates
  orchestration/            # Misrouting detection, switching, escalation
    misrouting_detector.py
    misrouting_rules.py
    switch_orchestrator.py
    transition_rules.py
    escalation_policy.py
    escalation_rules.py
    output_contract.py
  execution/                # LLM execution pipeline
    executor.py
    direct_execution.py
    repair_policy.py
  runtime/                  # Top-level runtime, planner, state management
    __init__.py             # CognitiveRouterRuntime
    planner.py              # RuntimePlanner
    session_runtime.py      # Orchestration loop
    state_updater.py        # State mutation helpers
    restore.py              # State deserialization
  llm/                      # LLM client abstraction
    model_client.py         # ModelClient Protocol
    ollama_client.py        # Ollama implementation
  prompts.py                # Prompt construction for regime execution
  validation.py             # Output validation (structural + semantic)
  state.py                  # RouterState, Handoff, SessionRecord
  storage.py                # JSON session persistence
  settings.py               # CLI settings
  cli.py                    # CLI entry point
  evolution/                # Revision engine (experimental)
    revision_engine.py
  embeddings.py             # DEPRECATED — embedding-based scoring
tests/                      # Pytest test suite
```

## Build and test

```bash
# Run full test suite
pytest

# Run specific test file
pytest tests/test_composer.py -v

# Run tests matching a keyword
pytest -k "orchestration" -v

# No special setup needed — no database, no network, no docker
```

## Code conventions

- Type hints on all function signatures and return types.
- Dataclasses for structured data. Frozen dataclasses for immutable values.
- `Protocol` classes for dependency injection (see `router/llm/model_client.py`).
- No wildcard imports except `__init__.py` re-exports.
- f-strings, not `.format()`.
- Imports: stdlib first, then package-relative imports. No third-party imports in core modules except `sentence_transformers` in `embeddings.py`.

## Testing conventions

- All LLM calls are mocked via the `ModelClient` protocol. Tests never make real network calls.
- Test files mirror source: `tests/test_routing.py` tests `router/routing.py`.
- Use `unittest.mock.MagicMock` and `unittest.mock.patch` for mocking.
- Assertions use plain `assert`, not `unittest.TestCase` methods.
- Test data is inline in test functions, not in fixture files.

## Architecture boundaries

These modules are stable and should NOT be modified unless the task explicitly says to:

- **Data models** (`router/models.py`): LIBRARY, LinePrimitive, Regime, Stage, all dataclasses. This is the system's vocabulary.
- **Grammar composer** (`router/routing/grammar_composer.py`, `grammar_rules.py`, `failure_selection.py`): Regime composition from the control line library. Grammar-driven, well-tested.
- **Orchestration** (`router/orchestration/`): Misrouting detection, switch orchestration, escalation policy, transition rules. These consume routing decisions but don't produce them.
- **Validation** (`router/validation.py`): Structural and semantic output validation. Stage-aware, profile-configurable.
- **Prompts** (`router/prompts.py`): System and user prompt construction for regime execution.
- **Execution** (`router/execution/`): LLM call, repair loop, direct execution bypass.
- **State** (`router/state.py`): RouterState, Handoff, serialization/deserialization.

## Active architecture migration

The routing system is migrating from deterministic lexical/embedding scoring to an LLM-proposer-first architecture. Files marked DEPRECATED above are being removed from the active routing path. The TaskAnalyzer in `analyzer.py` is becoming the primary routing proposer. Structural features from `feature_extraction.py` survive as corroboration evidence.

When working on routing-related tasks:
- The `Router.route()` method is being replaced. Do not add new scoring logic to it.
- The `TaskAnalyzer.propose_route()` method is the new primary path.
- Structural signals from `extract_routing_features()` are still used as validation evidence, not as scoring inputs.

## Key design concepts

- **Six regimes**: exploration, synthesis, epistemic, adversarial, operator, builder. Each has a dominant line, suppression lines, shape lines, and an optional tail (gate/transfer).
- **Control lines**: Atomic behavioral instructions from `LIBRARY` in `models.py`. Each has a function type (dominance, suppression, shape, gate, transfer) and compatibility/incompatibility constraints.
- **Regime composition**: The grammar composer selects lines based on stage, risk profile, and failure cost ranking, then validates the result against grammar rules (max 5 lines, 1-2 dominance, 1-2 suppression, 0-2 shape, 0-1 tail, no hard conflicts).
- **Output contract**: LLM output must be valid JSON with regime, purpose, artifact_type, artifact, completion_signal, failure_signal, recommended_next_regime. The artifact must contain stage-specific fields.

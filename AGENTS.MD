# AGENTS.md

## Project overview

The Cognitive Router is a Python runtime that selects and sequences reasoning regimes based on the current bottleneck. It routes by bottleneck (not topic), uses one dominant regime at a time, switches on failure profile, and preserves state through typed handoffs.

There are six regimes: Exploration, Synthesis, Epistemic, Adversarial, Operator, Builder.

The architecture is analyzer-led and mostly deterministic after planning. The safety valve is explicit fallback to Exploration when the active frame becomes invalid.

## Key architectural constraints

- Route by bottleneck, not by domain label.
- One dominant regime at a time unless mixed control is explicitly justified.
- Minimize switches — regime changes are costly.
- Preserve contradictions, assumptions, and unresolved questions across switches.
- External scaffolding controls LLM behavior; do not rely on the model to self-verify.
- The TaskAnalyzer is the primary proposer (v2 architecture). Router.route() is the feature-led fallback path.

## Repository layout

- `router/routing.py` — Router class, top-level route() method
- `router/routing/feature_extraction.py` — RoutingFeatures extraction from task text
- `router/analyzer.py` — TaskAnalyzer, decision_from_analysis
- `router/models.py` — RoutingDecision, RoutingFeatures, Regime, Stage, and all typed models
- `router/runtime/planner.py` — Planner orchestration
- `router/runtime/session_runtime.py` — Session loop, bounded orchestration
- `router/runtime/__init__.py` — CognitiveRouterRuntime
- `router/orchestration/switch_orchestrator.py` — Switch decisions
- `router/orchestration/transition_rules.py` — Transition conditions including collapse detection
- `router/orchestration/stop_policy.py` — Stop/continue decisions
- `router/orchestration/misrouting_detector.py` — Misrouting signal detection
- `router/composition/regime_composer.py` — RegimeComposer
- `router/state.py` — RouterState, state update functions

## Testing
```bash
# Run focused tests
pytest tests/ -k "test_name_pattern" -x -v

# Run a specific test module
pytest tests/test_module.py -x -v

# Run full suite
pytest tests/ -x -v
```

Always run the smallest relevant test subset first. Only run the full suite if focused tests pass.

## Code style

- Python 3.10+, type hints on all public methods
- Prefer explicit control logic over inferred heuristics
- Define small explicit vocabularies rather than inventing ad-hoc strings
- Keep test names descriptive and coupled to the behavior they verify
- No prompt rewrites when the problem is control logic
- No repo-wide redesigns — scope changes to the stated objective

## What not to do

- Do not rewrite handoff/state serialization code unless a task directly requires it
- Do not introduce new regime types
- Do not change the model provider configuration (dolphin29:latest via Ollama, OpenAI API routing layer)
- Do not add dependencies without explicit justification
- Do not combine unrelated concerns in a single change

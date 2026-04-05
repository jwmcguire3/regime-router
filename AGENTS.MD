# AGENTS.md

## Project overview

The Cognitive Router is a staged execution runtime that controls how an LLM approaches a task under an external control plane.

It does not simply prompt once and trust the answer. The runtime classifies task shape, extracts deterministic features, gets analyzer-led route proposals when enabled, composes a regime from typed primitives, executes under a structured output contract, validates externally, updates canonical state, computes forward handoff, and may continue through a bounded orchestration loop.

The six stages are:
- exploration
- synthesis
- epistemic
- adversarial
- operator
- builder

The runtime owns the control plane. The model produces proposals and artifacts, but the runtime decides the active regime, validates outputs, records state, computes handoffs, and decides whether to stop or continue.

## Current repo reality that matters for these tasks

- The runtime and settings now default to the DeepSeek provider, not Ollama.
- OpenAI-compatible provider handling is shared between `openai` and `deepseek`.
- `Router.route()` still contains placeholder feature-led routing behavior.
- The orchestration loop currently evaluates stop policy before switch logic.
- Handoff support is already materially present:
  - `Handoff` already includes stable, tentative, broken, and do-not-relitigate elements
  - downstream prompts already include Prior Stage Context
- The current gap is not “missing handoff.” The gap is that handoff influence, completion semantics, and bad-output recovery are not yet strong enough.

## Repository layout

Core runtime and orchestration:
- `router/runtime/__init__.py` — `CognitiveRouterRuntime`, runtime wiring, provider-aware client creation
- `router/runtime/planner.py` — deterministic planning, analyzer-led vs feature-led routing path, fast-path gate
- `router/runtime/session_runtime.py` — bounded orchestration loop, stop-before-switch behavior, loop prevention
- `router/runtime/state_updater.py` — `build_router_state`, `update_router_state_from_execution`, `compute_forward_handoff`, extraction helpers
- `router/orchestration/stop_policy.py` — stop decisions and builder gating
- `router/orchestration/switch_orchestrator.py` — switch recommendation and next-stage resolution
- `router/orchestration/transition_rules.py` — stage-to-stage transition logic
- `router/orchestration/misrouting_detector.py` — deterministic post-stage misrouting signals

Routing and composition:
- `router/routing.py` — top-level `Router.route()` placeholder path
- `router/routing/feature_extraction.py` — deterministic routing feature extraction
- `router/routing/composer.py` — `RegimeComposer`
- `router/routing/grammar_composer.py` — grammar-based regime composition

Prompts, validation, execution:
- `router/prompts.py` — stage-aware system, user, and repair prompts
- `router/validation.py` — external output validation
- `router/execution/executor.py` — single-stage execution + repair loop
- `router/execution/direct_execution.py` — direct passthrough execution path

Models and state:
- `router/models.py` — typed models, stage enums, routing decision, execution result, artifact hints/fields
- `router/state.py` — `RouterState`, `Handoff`, serialization and restoration helpers
- `router/analyzer.py` — analyzer output parsing and analyzer-led `RoutingDecision` construction

## Testing

Run the smallest relevant test subset first. Only broaden after focused tests pass.

Examples:
```bash
pytest tests/test_stop_policy.py -x -v
pytest tests/test_session_runtime.py -x -v
pytest tests/test_prompts.py -x -v
pytest tests/test_runtime_state_updater.py -x -v
pytest tests/ -k "artifact_aware or invalid_output or handoff" -x -v
pytest tests/ -x -v
```

## Code style and implementation rules

- Python 3.10+, type hints on public methods
- Prefer explicit control logic over soft prompt heuristics
- Keep new vocabularies compact and normalized
- Avoid repo-wide redesigns
- Keep changes scoped to the task objective
- Preserve existing validation structure unless the task directly requires a small extension
- Preserve truthful bookkeeping: completion, failure, switch cause, and stop reason should remain inspectable and specific

## What not to do

- Do not change provider defaults or provider configuration behavior
- Do not rewrite analyzer behavior unless a task directly requires it
- Do not rewrite routing feature extraction unless the task directly requires it
- Do not replace the existing validation/repair architecture with a new framework
- Do not delete or flatten existing handoff fields
- Do not solve a control-layer gap only by adding prompt wording
- Do not create parallel test architectures when extending existing focused test modules is sufficient

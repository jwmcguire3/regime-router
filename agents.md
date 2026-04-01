# Cognitive Router — AGENTS.md

## Project
Python 3.10+ runtime for controlling LLM reasoning regime selection and execution.
334+ pytest tests. All changes must preserve passing tests unless a test is explicitly
identified for removal or modification in the task prompt.

## Architecture
- `router/runtime/` — main runtime, planner, session
- `router/classifier.py` — task classifier (being demoted, not removed)
- `router/analyzer.py` — task analyzer (primary route proposer)
- `router/routing/` — composer, grammar_composer
- `router/execution/` — executor, direct_execution, repair_policy
- `router/validation.py` — output validator
- `router/orchestration/` — misrouting, switching, escalation, transitions
- `router/state.py` — RouterState, Handoff, RegimeStep, SwitchDecisionRecord
- `router/models.py` — line primitives, stage maps, canonical maps
- `router/prompts.py` — prompt builder

## Test Commands
```bash
pytest                          # full suite
pytest -x                       # stop on first failure
pytest -k "test_classifier"     # run classifier tests only
pytest -k "test_analyzer"       # run analyzer tests only
pytest --tb=short               # short tracebacks
```

## Constraints
- Never delete a test without explicit instruction in the task prompt.
- Never change the six-stage model (exploration, synthesis, epistemic, adversarial, operator, builder).
- Never modify the line primitive library or its IDs.
- Preserve all existing public method signatures unless the task explicitly calls for signature changes.
- When adding new fields to dataclasses/models, always provide defaults so existing code doesn't break.
- Run `pytest` after every change and fix what you broke before finishing.

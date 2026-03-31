# Cognitive Router

## Build & Test
- Python 3.10+
- Run tests: `pytest -x -q` (stop on first failure)
- Run specific: `pytest tests/test_composer.py -x -q`
- All 334+ tests must pass after every change

## Architecture Rules
- NEVER modify test files without explicit instruction
- NEVER change the public interface of existing classes without updating all call sites
- Commit after each logical change with a descriptive message
- Run pytest before and after every file modification

## Module Structure
- `router/models.py` — all data models, LIBRARY, stage constants
- `router/routing/composer.py` — RegimeComposer (being replaced)
- `router/routing/` — scoring, confidence, feature extraction
- `router/orchestration/` — misrouting, escalation, switch orchestrator
- `router/state.py` — RouterState, serialization
- `router/validation.py` — OutputValidator

## Key Invariants
- RegimeComposer.compose(stage, risk_profile, handoff_expected) -> Regime
- This signature is called from: routing.py, state.py, runtime/, orchestration/
- The Regime dataclass structure must not change
- LinePrimitive, LIBRARY, and all Stage enums are in models.py
- Every composed Regime must have exactly 1 dominance line
- Total lines per regime <= 5

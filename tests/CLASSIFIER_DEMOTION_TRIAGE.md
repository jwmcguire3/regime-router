# TaskClassifier Demotion Triage (Audit Only)

## Scope
Audit target: demote `TaskClassifier` from primary routing gate to advisory signal.

## Collection summary
- `pytest --collect-only -q`
- **Total tests collected:** 837

## Search summary
- `route_type="direct"` assertions: found in `tests/test_classifier.py` (4 tests).
- `route_type="regime"` assertions: found in `tests/test_classifier.py` (5 tests).
- Mock/patch of `TaskClassifier.classify` to force routing flow: **none found**.
- Action-verb + artifact-noun pattern matching tests: `tests/test_classifier.py`.
- Tests asserting direct execution bypasses analyzer entirely: **none found**.

---

## REMOVE (0)
Tests that assert classifier-as-gate behavior and should be deleted.

- None identified.

## MODIFY (0)
Tests that test classifier logic but need updated assertions.

- None identified.

## KEEP (9)
Tests that validate classifier pattern extraction in isolation and remain valid as signal tests after demotion.

1. `tests/test_classifier.py::test_classifies_write_breakout_game_code_as_direct`
2. `tests/test_classifier.py::test_classifies_build_rest_api_as_direct`
3. `tests/test_classifier.py::test_classifies_draft_q3_email_as_direct`
4. `tests/test_classifier.py::test_classifies_fragments_spine_statement_as_regime`
5. `tests/test_classifier.py::test_classifies_stress_test_frame_as_regime`
6. `tests/test_classifier.py::test_classifies_pricing_decision_question_as_regime`
7. `tests/test_classifier.py::test_classifies_explore_interpretations_as_regime`
8. `tests/test_classifier.py::test_classifies_fix_bug_as_direct`
9. `tests/test_classifier.py::test_classifies_architecture_fragility_why_question_as_regime`

## REWRITE (0)
Tests that validate overall routing flow and must be updated for analyzer-first flow.

- None identified that currently depend on classifier gating behavior.

---

## Category totals
- REMOVE: **0**
- MODIFY: **0**
- KEEP: **9**
- REWRITE: **0**

## Notes
- Current suite appears to already test analyzer-led stage routing via `TaskAnalyzer`-mocked flow in routing tests, rather than classifier gating.
- No test currently patches `TaskClassifier.classify` to control end-to-end routing, and no test currently asserts analyzer bypass due to classifier-direct gate.

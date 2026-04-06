# Mypy error diagnosis (what each value should be)

This maps the reported mypy errors to the concrete types mypy expects at each location.

## 1) `router/settings.py`

- `settings.py:58` (`data["max_switches"] < 0`): `data` is currently inferred as `dict[str, object]`; `data["max_switches"]` should be `int`.
- `settings.py:60` (`return cls(**data)`): `data` should not be `dict[str, object]`; it should be a `TypedDict`/explicitly typed payload matching `UserSettings` fields:
  - `provider: str`
  - `model: str`
  - `openai_base_url: str`
  - `openai_api_key_env: str`
  - `use_task_analyzer: bool`
  - `task_analyzer_model: str`
  - `debug_routing: bool`
  - `bounded_orchestration: bool`
  - `max_switches: int`

## 2) `router/state.py`

- `state.py:224` (`asdict(obj)`): `obj` should be narrowed to a dataclass instance, not generic `object`.
- `state.py:272-277` (`SessionRecord(...)`): each `to_jsonable(...)` result should be cast/narrowed to `dict[str, object]` (and `router_state` to `dict[str, object] | None`) before passing into `SessionRecord`.
- `state.py:398-401` (`ReentryJustification(...)`): fields should be `str`; after runtime checks, use casts/typed locals so mypy sees `str` rather than `Any | None`.
- `state.py:504-508` (`PolicyEvent(...)`): fields should be `str`; after `isinstance(..., str)` checks, store into typed locals before constructing.

## 3) `router/orchestration/misrouting_detector.py` and `switch_orchestrator.py`

- `RegimeComposer` imports from `router.routing` are seen as runtime variables (because of dynamic import in `router/routing.py`), not valid static class types.
- Constructor params (`composer`) should be typed as one of:
  - concrete class from a statically importable module (`router.routing.composer.RegimeComposer`) **or**
  - a `Protocol` exposing `compose(...)`.
- `.compose` attribute errors are the same root cause: the type should be a known class/protocol with `compose`.

## 4) `router/orchestration/transition_rules.py`

- `transition_rules.py:71` (`control_failures` iteration): `validation.get(...)` is inferred as `object`; this should be narrowed to `Iterable[object]`/`list[object]` before iterating.

## 5) `router/validation.py`

- `validation.py:276`, `308`: `cfg[...]` entries passed to `int(...)` are currently `object`; they should be `int | str` (or narrowed via helper getters).
- `validation.py:285`: `cfg[...]` entry passed to `float(...)` should be `float | str` after narrowing.

## 6) `router/prompts.py`

- `prompts.py:345-346` and `401-402`: `validation.get("semantic_failures", [])` is inferred as `object`; should be narrowed to `list[str]` (or `Iterable[object]` and converted to `list[str]`) before iteration and `_failed_fields(...)` call.

## 7) `router/runtime/state_updater.py`

- `state_updater.py:145`, `450`: `validation.get("semantic_failures", [])` is inferred as `object`; should be normalized to `list[str]` before list comprehension/iteration.

## 8) `router/runtime/session_runtime.py`

- `session_runtime.py:192`, `200`, `217`: `orchestrated.next_regime` is `Regime | None`; access to `.stage` should happen only after explicit `if orchestrated.next_regime is not None` narrowing.
- `session_runtime.py:230`: assignment target should allow `Regime | None` or be guarded so assigned value is definitely `Regime`.

## 9) `router/execution/repair_policy.py`

- `repair_policy.py:22`: `validation.get("semantic_failures", [])` should be narrowed to iterable/list before iteration.

## 10) `router/cli.py`

- `cli.py:73`: `_format_stage_contributions` parameter should accept `Mapping[Stage, list[str]]` (or generic `Mapping[Hashable, list[str]]`) to match caller.
- `cli.py:242`, `420`: `_resolve_setting(...)` returns `object`; the value passed to `int(...)` should be narrowed to `int | str` first.
- `cli.py:263`: `CognitiveRouterRuntime.__init__` currently has no `model_profile` kwarg; either add `model_profile: str = "strict"` to runtime ctor or remove that keyword from call.

## 11) Test-only typing mismatches (`tests/...`)

### A) Missing local variable annotations
- `Need type annotation for ...` errors (`init_calls`, `execute_calls`, `plan_calls`, `parsed`) should be explicit, e.g. `list[tuple[...]]`, `dict[str, object]`, etc.

### B) Method monkeypatching
- `Cannot assign to a method` in tests should use:
  - `monkeypatch.setattr(...)`, or
  - assign on instance with `cast(Any, obj).method = ...` (test-only), or
  - injectable dependency seams.

### C) Stub class compatibility
- `_NoopDetector`, `_NoopEscalation`, `_StaticOrchestrator`, `StubAnalyzer`, `FixedDecisionAnalyzer`, `_FixedDecisionAnalyzer` should satisfy the exact protocol/base class expected by runtime/planner call sites:
  - either subclass real classes,
  - or adopt `Protocol`-typed parameters in production code.

### D) `RuntimePlanner.plan(**kwargs)` tests (`test_runtime_planner_determinism.py`)
- The kwargs dict is too loosely typed (`dict[str, ... union ...]`), so unpacking into `plan(...)` fails.
- The kwargs object should be a `TypedDict` with exact `plan` parameters:
  - `bottleneck: str`
  - `router_state: RouterState | None`
  - `use_task_analyzer: bool`
  - `task_analyzer: TaskAnalyzer | None`
  - `risk_profile: set[str] | None`
  - `handoff_expected: bool`
  - `task_signals: list[str] | None`
  - `risks_inferred: bool`
  - `analyzer_result: TaskAnalyzerOutput | None`

### E) `SimpleNamespace` vs concrete model type
- `test_fallback_regressions.py:115`: `routing_features` argument should be real `RoutingFeatures` instance (or code should accept a protocol/Mapping).

### F) Specific argument type mismatches
- `test_openai_client.py:78`: `HTTPError(..., hdrs=...)` expects `Message[str, str]`, not `None`.
- `test_regime_field_validation.py:17`: assigned value should be `list[str]`, not `str`.

## 12) Root-cause summary

Most errors come from four recurring patterns:

1. `dict.get(...)` on `dict[str, object]` without narrowing, then using result as concrete numeric/list type.
2. Dynamic import aliasing (`RegimeComposer = _module.RegimeComposer`) that mypy cannot treat as a type.
3. Optional values (`Regime | None`) used without `None`-checks.
4. Tests using dynamic monkeypatching or loosely typed kwargs without explicit test-time annotations/casts/protocol alignment.

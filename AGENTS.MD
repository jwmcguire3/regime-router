# AGENTS.md

## Purpose
This repository is a provider-aware cognitive router with:
- persisted CLI settings
- a Python CLI entrypoint
- a PowerShell wrapper
- provider-specific runtime client creation
- JSON run persistence
- wrapper and CLI tests

The current task is to add DeepSeek as a first-class provider while preserving existing OpenAI and Ollama behavior.

## Goal
Implement first-class `deepseek` provider support across settings, CLI, runtime, wrapper, and tests.

Platform default after this change:
- provider: `deepseek`
- model: `deepseek-reasoner`
- base URL: `https://api.deepseek.com`
- API key env var: `DEEPSEEK_API_KEY`

DeepSeek should reuse the existing OpenAI-compatible HTTP client rather than introducing a separate provider client unless a real compatibility issue requires it.

## Required behavior
1. `deepseek` must be accepted anywhere provider selection/validation currently accepts `ollama` and `openai`.
2. Fresh/default settings should resolve to DeepSeek Reasoner.
3. Existing settings files should remain backward compatible.
4. Switching providers should preserve custom models where appropriate and swap defaults where the current model is still the previous provider default.
5. Runtime provider resolution must not silently fall back to Ollama for unknown providers.
6. Wrapper first-run initialization must align with Python-side defaults.
7. User-facing text should stop implying that all compatible-provider settings are OpenAI-only when that is no longer true.

## Provider semantics
- `ollama` -> local Ollama client
- `openai` -> OpenAI-compatible client targeting actual OpenAI defaults
- `deepseek` -> OpenAI-compatible client targeting DeepSeek defaults

Do not fake DeepSeek by keeping `provider="openai"` internally. It must be a real provider option.

## Files likely involved
- `router/settings.py`
- `router/cli.py`
- `router/runtime/__init__.py`
- `router.ps1`
- `tests/test_router_wrapper.py`
- any existing settings/runtime tests

## Constraints
- Preserve backward compatibility for persisted settings JSON.
- Keep changes small and targeted.
- Do not introduce a new DeepSeek client class unless required by a demonstrated compatibility gap.
- Prefer explicit failure for unsupported providers.
- Do not change unrelated routing/orchestration logic.
- Do not rewrite large sections of the codebase just to rename config fields.
- If field names like `openai_base_url` remain for compatibility, update help text/comments to clarify they are used for OpenAI-compatible providers.

## Done when
- `deepseek` is selectable in settings and wrapper menus
- fresh defaults are DeepSeek Reasoner
- runtime instantiates the OpenAI-compatible client for DeepSeek
- tests cover DeepSeek defaults/provider transitions
- no silent fallback routes `deepseek` to Ollama
- OpenAI and Ollama support remain intact

## Validation expectations
At minimum, run the relevant test subset for modified files.
If practical, also perform a live or mocked smoke check for:
- `deepseek-reasoner` chat completion path
- model listing against DeepSeek base URL

## Implementation guidance
- Keep edits surgical.
- Prefer extending existing provider/default helpers over adding new abstraction layers.
- Add comments only where the intent would otherwise be unclear.
- If a runtime/API incompatibility is found, surface the exact failure and patch only that incompatibility.

## Test discipline
Every behavior change must be accompanied by tests in the same task unless the code path is impossible to test directly.

For each task:
1. identify the affected test surface
2. add or update tests first or alongside the implementation
3. run the smallest relevant test subset for the touched area
4. confirm adjacent existing tests still pass
5. do not mark the task done without reporting what was run

Do not defer all testing to the final integration pass.

## Minimum validation rule
For any touched file:
- run focused tests for that behavior
- run any pre-existing tests that cover adjacent behavior
- after merge, run the full existing test suite

## Reporting rule
When finishing a task, report:
- files changed
- tests added/updated
- commands run
- pass/fail result
- any remaining untested risk

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Set

from .models import RoutingDecision, RoutingFeatures
from .routing import Router, extract_routing_features, infer_risk_profile
from .runtime import CognitiveRouterRuntime
from .settings import CliSettings, CliSettingsStore
from .state import Handoff, make_record
from .storage import SessionStore


class CliOutputFormatter:
    def __init__(self, mode: str = "verbose") -> None:
        self.mode = mode

    @property
    def compact(self) -> bool:
        return self.mode == "compact"

    def print_section(self, title: str) -> None:
        print()
        print(f"=== {title} ===")

    def print_kv(self, key: str, value: object) -> None:
        print(f"{key:<28}: {value}")

    def print_block(self, title: str, body: str) -> None:
        self.print_section(title)
        print(body)


def print_routing(decision: RoutingDecision, fmt: CliOutputFormatter) -> None:
    fmt.print_section("Routing summary")
    fmt.print_kv("Current bottleneck", decision.bottleneck)
    fmt.print_kv("Primary regime", decision.primary_regime.value)
    fmt.print_kv("Runner-up regime", decision.runner_up_regime.value if decision.runner_up_regime else "none")
    fmt.print_kv("Confidence level", decision.confidence.level)
    if not fmt.compact:
        fmt.print_kv(
            "Confidence scores",
            f"top={decision.confidence.top_stage_score}, runner-up={decision.confidence.runner_up_score}, gap={decision.confidence.score_gap}",
        )
        fmt.print_kv("Confidence rationale", decision.confidence.rationale)
    fmt.print_kv("Deterministic scores", decision.deterministic_score_summary or "n/a")
    if decision.deterministic_score_contributions and not fmt.compact:
        fmt.print_kv(
            "Deterministic contributions",
            Router._format_stage_contributions(decision.deterministic_score_contributions),
        )
    fmt.print_kv("Analyzer enabled", decision.analyzer_enabled)
    fmt.print_kv(
        "Analyzer used",
        f"{decision.analyzer_used} (changed primary={decision.analyzer_changed_primary}, changed runner-up={decision.analyzer_changed_runner_up})",
    )
    if decision.analyzer_summary:
        fmt.print_kv("Analyzer summary", decision.analyzer_summary)
    fmt.print_kv("Why primary wins now", decision.why_primary_wins_now)
    fmt.print_kv("Switch trigger", decision.switch_trigger)


def print_routing_debug(
    *,
    decision: RoutingDecision,
    features: RoutingFeatures,
    signals: List[str],
    risks: Set[str],
    fmt: CliOutputFormatter,
) -> None:
    fmt.print_section("Debug")
    fmt.print_kv("Structural signals", signals or [])
    fmt.print_kv("Risk profile", sorted(risks))
    fmt.print_kv(
        "Feature pressures",
        f"decision={features.decision_pressure}, evidence={features.evidence_demand}, fragility={features.fragility_pressure}, recurrence={features.recurrence_potential}, possibility={features.possibility_space_need}",
    )
    if not fmt.compact:
        fmt.print_kv("Detected markers", json.dumps(features.detected_markers, ensure_ascii=False))
        fmt.print_kv(
            "Confidence detail",
            f"level={decision.confidence.level}, rationale={decision.confidence.rationale}, nontrivial_stage_count={decision.confidence.nontrivial_stage_count}, weak_lexical_dependence={decision.confidence.weak_lexical_dependence}, structural_feature_state={decision.confidence.structural_feature_state}",
        )
        fmt.print_kv("Stage scores", decision.deterministic_score_summary or "n/a")
        fmt.print_kv("Stage contributions", Router._format_stage_contributions(decision.deterministic_score_contributions))
    fmt.print_kv(
        "Analyzer state",
        f"enabled={decision.analyzer_enabled}, used={decision.analyzer_used}, summary={decision.analyzer_summary or 'n/a'}",
    )


def print_handoff(handoff: Handoff, fmt: CliOutputFormatter) -> None:
    fmt.print_section("Handoff")
    fmt.print_kv("Current bottleneck", handoff.current_bottleneck)
    fmt.print_kv("Dominant frame", handoff.dominant_frame)
    fmt.print_kv("What is known", ", ".join(handoff.what_is_known))
    fmt.print_kv("What remains uncertain", ", ".join(handoff.what_remains_uncertain))
    if not fmt.compact:
        fmt.print_kv("Active contradictions", ", ".join(handoff.active_contradictions))
        fmt.print_kv("Assumptions in play", ", ".join(handoff.assumptions_in_play))
    fmt.print_kv("Main risk if continue", handoff.main_risk_if_continue)
    fmt.print_kv("Recommended next regime", handoff.recommended_next_regime.value if handoff.recommended_next_regime else "none")
    fmt.print_kv("Minimum useful artifact", handoff.minimum_useful_artifact)


def print_validation(validation: Dict[str, object], fmt: CliOutputFormatter) -> None:
    fmt.print_section("Validation")
    for k, v in validation.items():
        if k == "parsed":
            continue
        fmt.print_kv(str(k), v)


def parse_risk_profile(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _resolve_setting(override: Optional[object], stored: object) -> object:
    return stored if override is None else override


def _resolved_cli_settings(args: argparse.Namespace) -> CliSettings:
    store = CliSettingsStore(path=args.settings_file)
    stored = store.load()
    settings = CliSettings(
        model=str(_resolve_setting(getattr(args, "model", None), stored.model)),
        use_task_analyzer=bool(_resolve_setting(getattr(args, "use_task_analyzer", None), stored.use_task_analyzer)),
        task_analyzer_model=str(
            _resolve_setting(getattr(args, "task_analyzer_model", None), stored.task_analyzer_model)
        ),
        debug_routing=bool(_resolve_setting(getattr(args, "debug_routing", None), stored.debug_routing)),
        bounded_orchestration=bool(
            _resolve_setting(getattr(args, "bounded_orchestration", None), stored.bounded_orchestration)
        ),
        max_switches=int(_resolve_setting(getattr(args, "max_switches", None), stored.max_switches)),
    )
    if settings.max_switches < 0:
        raise ValueError("--max-switches must be >= 0")
    return settings


def cmd_run(args: argparse.Namespace) -> int:
    settings = _resolved_cli_settings(args)
    fmt = CliOutputFormatter(args.output)
    runtime = CognitiveRouterRuntime(
        ollama_base_url=args.base_url,
        use_task_analyzer=settings.use_task_analyzer,
        task_analyzer_model=settings.task_analyzer_model,
    )
    store = SessionStore(root=args.out_dir)
    risk_profile = parse_risk_profile(args.risks)

    decision, regime, result, handoff = runtime.execute(
        task=args.task,
        model=settings.model,
        risk_profile=risk_profile,
        handoff_expected=not args.no_handoff,
        bounded_orchestration=settings.bounded_orchestration,
        max_switches=settings.max_switches,
    )

    print_routing(decision, fmt)
    fmt.print_block("Regime output", regime.render())
    fmt.print_block("Model output", result.raw_response)
    print_validation(result.validation, fmt)
    print_handoff(handoff, fmt)

    record = make_record(args.task, risk_profile, settings.model, decision, regime, result, handoff, runtime.router_state)
    path = store.save(record, filename=args.save_as)
    print(f"Saved run to: {path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    settings = _resolved_cli_settings(args)
    fmt = CliOutputFormatter(args.output)
    runtime = CognitiveRouterRuntime(
        ollama_base_url=args.base_url,
        use_task_analyzer=settings.use_task_analyzer,
        task_analyzer_model=settings.task_analyzer_model,
    )
    decision, regime, handoff = runtime.plan(
        bottleneck=args.task,
        risk_profile=parse_risk_profile(args.risks),
        handoff_expected=not args.no_handoff,
    )
    print_routing(decision, fmt)
    if settings.debug_routing:
        features = extract_routing_features(args.task)
        signals = features.structural_signals
        risks = infer_risk_profile(args.task, parse_risk_profile(args.risks))
        print_routing_debug(decision=decision, features=features, signals=signals, risks=risks, fmt=fmt)
    fmt.print_block("Regime output", regime.render())
    print_handoff(handoff, fmt)
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    store = SessionStore(root=args.out_dir)
    runs = store.list_runs()
    if not runs:
        print("No saved runs found.")
        return 0
    for run in runs:
        print(run)
    return 0


def cmd_show_run(args: argparse.Namespace) -> int:
    store = SessionStore(root=args.out_dir)
    data = store.load(args.filename)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    runtime = CognitiveRouterRuntime(ollama_base_url=args.base_url)
    models = runtime.ollama.list_models()
    print(json.dumps(models, indent=2, ensure_ascii=False))
    return 0


def cmd_settings_show(args: argparse.Namespace) -> int:
    store = CliSettingsStore(path=args.settings_file)
    settings = store.load()
    payload = {
        "settings_file": str(store.path),
        "settings": settings.to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_settings_set(args: argparse.Namespace) -> int:
    store = CliSettingsStore(path=args.settings_file)
    current = store.load()
    updated = CliSettings(
        model=str(_resolve_setting(args.model, current.model)),
        use_task_analyzer=bool(_resolve_setting(args.use_task_analyzer, current.use_task_analyzer)),
        task_analyzer_model=str(_resolve_setting(args.task_analyzer_model, current.task_analyzer_model)),
        debug_routing=bool(_resolve_setting(args.debug_routing, current.debug_routing)),
        bounded_orchestration=bool(_resolve_setting(args.bounded_orchestration, current.bounded_orchestration)),
        max_switches=int(_resolve_setting(args.max_switches, current.max_switches)),
    )
    if updated.max_switches < 0:
        raise ValueError("--max-switches must be >= 0")
    path = store.save(updated)
    payload = {
        "settings_file": str(path),
        "settings": updated.to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_settings_reset(args: argparse.Namespace) -> int:
    store = CliSettingsStore(path=args.settings_file)
    settings = store.reset()
    payload = {
        "settings_file": str(store.path),
        "settings": settings.to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cognitive router prototype with Ollama-backed execution and JSON persistence."
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--out-dir", default="runs", help="Directory for saved JSON runs")
    parser.add_argument("--settings-file", default=".router_settings.json", help="Path to persisted CLI settings JSON")
    parser.add_argument(
        "--output",
        choices=["verbose", "compact"],
        default="verbose",
        help="Console output style for plan/run commands",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Route + compose + execute against Ollama + save JSON")
    run_p.add_argument("--task", required=True, help="Task or bottleneck description")
    run_p.add_argument("--model", default=None, help="Ollama model name")
    run_p.add_argument("--risks", default="", help="Comma-separated risk profile tags")
    run_p.add_argument("--save-as", default=None, help="Optional output JSON filename")
    run_p.add_argument("--no-handoff", action="store_true", help="Disable tail/transfer line where optional")
    run_p.add_argument(
        "--use-task-analyzer",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable optional LLM task analyzer for low-confidence routing cases",
    )
    run_p.add_argument("--task-analyzer-model", default=None, help="Ollama model for task analyzer when enabled")
    run_p.add_argument(
        "--bounded-orchestration",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable bounded switch orchestration during execution",
    )
    run_p.add_argument("--max-switches", default=None, type=int, help="Maximum switches allowed in bounded orchestration")
    run_p.add_argument(
        "--debug-routing",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override debug routing setting (effective in plan output)",
    )
    run_p.set_defaults(func=cmd_run)

    plan_p = sub.add_parser("plan", help="Route + compose without calling Ollama")
    plan_p.add_argument("--task", required=True, help="Task or bottleneck description")
    plan_p.add_argument("--risks", default="", help="Comma-separated risk profile tags")
    plan_p.add_argument("--no-handoff", action="store_true", help="Disable tail/transfer line where optional")
    plan_p.add_argument(
        "--use-task-analyzer",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable optional LLM task analyzer for low-confidence routing cases",
    )
    plan_p.add_argument("--task-analyzer-model", default=None, help="Ollama model for task analyzer when enabled")
    plan_p.add_argument(
        "--debug-routing",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Print inspectable routing internals (features, scores, confidence, analyzer state)",
    )
    plan_p.add_argument("--model", default=None, help="Override persisted default model setting for compatibility")
    plan_p.add_argument(
        "--bounded-orchestration",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override bounded orchestration setting for compatibility",
    )
    plan_p.add_argument("--max-switches", default=None, type=int, help="Override max switches setting for compatibility")
    plan_p.set_defaults(func=cmd_plan)

    list_p = sub.add_parser("list-runs", help="List saved run files")
    list_p.set_defaults(func=cmd_list_runs)

    show_p = sub.add_parser("show-run", help="Print a saved run JSON")
    show_p.add_argument("filename", help="Filename inside the runs directory")
    show_p.set_defaults(func=cmd_show_run)

    models_p = sub.add_parser("models", help="List models from local Ollama")
    models_p.set_defaults(func=cmd_models)

    settings_p = sub.add_parser("settings", help="Show/update/reset persisted CLI defaults")
    settings_sub = settings_p.add_subparsers(dest="settings_command")

    settings_show_p = settings_sub.add_parser("show", help="Show current CLI settings")
    settings_show_p.set_defaults(func=cmd_settings_show)

    settings_set_p = settings_sub.add_parser("set", help="Update one or more CLI settings")
    settings_set_p.add_argument("--model", default=None, help="Default Ollama model")
    settings_set_p.add_argument(
        "--use-task-analyzer",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Default task analyzer toggle",
    )
    settings_set_p.add_argument("--task-analyzer-model", default=None, help="Default analyzer model")
    settings_set_p.add_argument(
        "--debug-routing",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Default debug routing toggle",
    )
    settings_set_p.add_argument(
        "--bounded-orchestration",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Default bounded orchestration toggle",
    )
    settings_set_p.add_argument("--max-switches", default=None, type=int, help="Default maximum switches")
    settings_set_p.set_defaults(func=cmd_settings_set)

    settings_reset_p = settings_sub.add_parser("reset", help="Reset CLI settings to defaults")
    settings_reset_p.set_defaults(func=cmd_settings_reset)

    settings_p.set_defaults(func=cmd_settings_show)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

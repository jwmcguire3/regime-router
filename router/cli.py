from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Set

from .models import RoutingDecision, RoutingFeatures
from .routing import Router, extract_routing_features, infer_risk_profile
from .runtime import CognitiveRouterRuntime
from .state import Handoff, make_record
from .storage import SessionStore


def print_routing(decision: RoutingDecision) -> None:
    print("ROUTING HEADER")
    print(f"- Current bottleneck: {decision.bottleneck}")
    print(f"- Primary regime: {decision.primary_regime.value}")
    print(f"- Runner-up regime: {decision.runner_up_regime.value if decision.runner_up_regime else 'none'}")
    print(
        f"- Confidence: {decision.confidence.level} "
        f"(top={decision.confidence.top_stage_score}, runner-up={decision.confidence.runner_up_score}, gap={decision.confidence.score_gap})"
    )
    print(f"- Confidence rationale: {decision.confidence.rationale}")
    print(f"- Deterministic scores: {decision.deterministic_score_summary or 'n/a'}")
    if decision.deterministic_score_contributions:
        print(f"- Deterministic contributions: {Router._format_stage_contributions(decision.deterministic_score_contributions)}")
    print(
        f"- Analyzer enabled: {decision.analyzer_enabled}"
    )
    print(
        f"- Analyzer used: {decision.analyzer_used} "
        f"(changed primary={decision.analyzer_changed_primary}, changed runner-up={decision.analyzer_changed_runner_up})"
    )
    if decision.analyzer_summary:
        print(f"- Analyzer summary: {decision.analyzer_summary}")
    print(f"- Why primary wins now: {decision.why_primary_wins_now}")
    print(f"- Switch trigger: {decision.switch_trigger}")
    print()


def print_routing_debug(
    *,
    decision: RoutingDecision,
    features: RoutingFeatures,
    signals: List[str],
    risks: Set[str],
) -> None:
    print("ROUTING DEBUG")
    print(f"- Structural signals: {signals or []}")
    print(f"- Risk profile: {sorted(risks)}")
    print(f"- Feature pressures: decision={features.decision_pressure}, evidence={features.evidence_demand}, fragility={features.fragility_pressure}, recurrence={features.recurrence_potential}, possibility={features.possibility_space_need}")
    print(f"- Detected markers: {json.dumps(features.detected_markers, ensure_ascii=False)}")
    print(
        f"- Confidence detail: level={decision.confidence.level}, rationale={decision.confidence.rationale}, "
        f"nontrivial_stage_count={decision.confidence.nontrivial_stage_count}, weak_lexical_dependence={decision.confidence.weak_lexical_dependence}, structural_feature_state={decision.confidence.structural_feature_state}"
    )
    print(f"- Stage scores: {decision.deterministic_score_summary or 'n/a'}")
    print(f"- Stage contributions: {Router._format_stage_contributions(decision.deterministic_score_contributions)}")
    print(
        f"- Analyzer state: enabled={decision.analyzer_enabled}, used={decision.analyzer_used}, "
        f"summary={decision.analyzer_summary or 'n/a'}"
    )
    print()


def print_handoff(handoff: Handoff) -> None:
    print("HANDOFF")
    print(f"- Current bottleneck: {handoff.current_bottleneck}")
    print(f"- Dominant frame: {handoff.dominant_frame}")
    print(f"- What is known: {', '.join(handoff.what_is_known)}")
    print(f"- What remains uncertain: {', '.join(handoff.what_remains_uncertain)}")
    print(f"- Active contradictions: {', '.join(handoff.active_contradictions)}")
    print(f"- Assumptions in play: {', '.join(handoff.assumptions_in_play)}")
    print(f"- Main risk if continue: {handoff.main_risk_if_continue}")
    print(f"- Recommended next regime: {handoff.recommended_next_regime.value if handoff.recommended_next_regime else 'none'}")
    print(f"- Minimum useful artifact: {handoff.minimum_useful_artifact}")
    print()


def print_validation(validation: Dict[str, object]) -> None:
    print("VALIDATION")
    for k, v in validation.items():
        if k == "parsed":
            continue
        print(f"- {k}: {v}")
    print()


def parse_risk_profile(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def cmd_run(args: argparse.Namespace) -> int:
    runtime = CognitiveRouterRuntime(
        ollama_base_url=args.base_url,
        use_task_analyzer=args.use_task_analyzer,
        task_analyzer_model=args.task_analyzer_model,
    )
    store = SessionStore(root=args.out_dir)
    risk_profile = parse_risk_profile(args.risks)

    decision, regime, result, handoff = runtime.execute(
        task=args.task,
        model=args.model,
        risk_profile=risk_profile,
        handoff_expected=not args.no_handoff,
    )

    print_routing(decision)
    print(regime.render())
    print()
    print("MODEL OUTPUT")
    print(result.raw_response)
    print()
    print_validation(result.validation)
    print_handoff(handoff)

    record = make_record(args.task, risk_profile, args.model, decision, regime, result, handoff)
    path = store.save(record, filename=args.save_as)
    print(f"Saved run to: {path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    runtime = CognitiveRouterRuntime(
        ollama_base_url=args.base_url,
        use_task_analyzer=args.use_task_analyzer,
        task_analyzer_model=args.task_analyzer_model,
    )
    decision, regime, handoff = runtime.plan(
        bottleneck=args.task,
        risk_profile=parse_risk_profile(args.risks),
        handoff_expected=not args.no_handoff,
    )
    print_routing(decision)
    if args.debug_routing:
        features = extract_routing_features(args.task)
        signals = features.structural_signals
        risks = infer_risk_profile(args.task, parse_risk_profile(args.risks))
        print_routing_debug(decision=decision, features=features, signals=signals, risks=risks)
    print(regime.render())
    print()
    print_handoff(handoff)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cognitive router prototype with Ollama-backed execution and JSON persistence."
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--out-dir", default="runs", help="Directory for saved JSON runs")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Route + compose + execute against Ollama + save JSON")
    run_p.add_argument("--task", required=True, help="Task or bottleneck description")
    run_p.add_argument("--model", default="llama3", help="Ollama model name")
    run_p.add_argument("--risks", default="", help="Comma-separated risk profile tags")
    run_p.add_argument("--save-as", default=None, help="Optional output JSON filename")
    run_p.add_argument("--no-handoff", action="store_true", help="Disable tail/transfer line where optional")
    run_p.add_argument("--use-task-analyzer", action="store_true", help="Enable optional LLM task analyzer for low-confidence routing cases")
    run_p.add_argument("--task-analyzer-model", default="llama3", help="Ollama model for task analyzer when enabled")
    run_p.set_defaults(func=cmd_run)

    plan_p = sub.add_parser("plan", help="Route + compose without calling Ollama")
    plan_p.add_argument("--task", required=True, help="Task or bottleneck description")
    plan_p.add_argument("--risks", default="", help="Comma-separated risk profile tags")
    plan_p.add_argument("--no-handoff", action="store_true", help="Disable tail/transfer line where optional")
    plan_p.add_argument("--use-task-analyzer", action="store_true", help="Enable optional LLM task analyzer for low-confidence routing cases")
    plan_p.add_argument("--task-analyzer-model", default="llama3", help="Ollama model for task analyzer when enabled")
    plan_p.add_argument("--debug-routing", action="store_true", help="Print inspectable routing internals (features, scores, confidence, analyzer state)")
    plan_p.set_defaults(func=cmd_plan)

    list_p = sub.add_parser("list-runs", help="List saved run files")
    list_p.set_defaults(func=cmd_list_runs)

    show_p = sub.add_parser("show-run", help="Print a saved run JSON")
    show_p.add_argument("filename", help="Filename inside the runs directory")
    show_p.set_defaults(func=cmd_show_run)

    models_p = sub.add_parser("models", help="List models from local Ollama")
    models_p.set_defaults(func=cmd_models)

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

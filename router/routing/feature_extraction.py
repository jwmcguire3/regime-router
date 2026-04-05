from __future__ import annotations

import re
from typing import Dict, List, Tuple

from router.models import (
    RoutingFeatures,
    STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL,
    STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED,
    STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED,
)


def contains_any(text: str, phrases: Tuple[str, ...]) -> List[str]:
    matches: List[str] = []
    for phrase in phrases:
        if " " in phrase:
            if phrase in text:
                matches.append(phrase)
            continue

        if re.search(rf"\b{re.escape(phrase)}\b", text):
            matches.append(phrase)

    return matches


def has_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def score_from_matches(*matches: List[str]) -> int:
    return min(10, sum(len(group) for group in matches))


def extract_routing_features(task: str) -> RoutingFeatures:
    text = task.lower().replace("’", "'")

    # Grouped deterministic pattern families, optimized for task-shape markers.
    expansion_words = ("expand", "expands", "expansion", "broadens", "gets bigger", "widens", "balloons")
    define_words = ("define", "defined", "definition", "specify", "specified", "scope", "frame")
    concrete_words = ("concrete", "specific", "instance", "version", "example", "implementation")
    too_small_words = ("too small", "small", "narrow", "shrinks", "feels tiny", "cramped", "thin slice")
    parts_words = ("fragment", "fragments", "pieces", "parts", "components")
    whole_words = ("whole", "spine", "core", "throughline", "center", "backbone", "organizing logic")
    missing_words = ("missed", "missing", "lost", "not seen", "not grasped", "not holding")
    understood_words = ("understood", "clear", "comprehensible", "makes sense", "legible")

    evidence_words = (
        "evidence",
        "support",
        "verify",
        "unknown",
        "unknowns",
        "unclear",
        "unresolved",
        "proof",
        "confidence",
    )
    uncertainty_words = ("uncertain", "ambigu", "not sure", "missing information", "what is missing")
    uncertainty_characterization_words = (
        "can't tell",
        "can't tell what kind",
        "don't know what kind",
        "hard to characterize",
        "can't characterize",
        "can't identify",
        "can't name it yet",
    )

    decision_words = (
        "decide",
        "deciding",
        "decision",
        "choose",
        "choosing",
        "commit",
        "recommend",
        "recommendation",
        "make a call",
        "what should we do",
        "next move",
        "time pressure",
        "ship now",
        "best option now",
        "now",
        "this week",
        "immediate",
        "select",
        "selecting",
    )
    tradeoff_words = ("tradeoff", "trade-off", "between options", "opportunity cost")

    fragility_words = (
        "fragile",
        "break",
        "stress test",
        "failure mode",
        "failure modes",
        "weakest points",
        "weak spots",
        "strongest objections",
        "vulnerabilities",
        "where this breaks",
        "break under pressure",
        "how this could fail",
        "attack this frame",
        "stress points",
        "risk",
        "destabil",
        "brittle",
    )
    launch_words = ("launch", "production", "deploy", "deployment", "go-live", "trust", "customer-facing")

    recurrence_words_strong = (
        "repeatable",
        "reusable",
        "template",
        "playbook",
        "systematize",
        "systematized",
        "systematizing",
        "standardize",
        "standardized",
    )
    recurrence_words_generic = ("pattern",)
    builder_words = ("productize", "modules", "interfaces", "workflow", "automation")

    possibility_words = (
        "possibility",
        "explore",
        "exploration",
        "brainstorm",
        "alternatives",
        "option space",
        "open",
        "multiple frames",
        "multiple possible frames",
        "multiple perspectives",
        "multiple interpretations",
        "perspectives",
        "interpretations",
        "map the space",
    )
    convergence_words = ("too early", "premature", "locked in", "single frame", "compresses", "narrowing")
    anti_convergence_words = (
        "keep it open",
        "rather than converging",
        "instead of converging",
        "delay convergence",
        "delaying convergence",
        "before narrowing",
    )
    negated_closure_words = (
        "do not decide",
        "don't decide",
        "not decide yet",
        "do not recommend",
        "don't recommend",
        "do not choose",
        "don't choose",
        "do not make a call",
        "not ready to decide",
    )

    matches: Dict[str, List[str]] = {}

    expansion_hits = contains_any(text, expansion_words)
    define_hits = contains_any(text, define_words)
    concrete_hits = contains_any(text, concrete_words)
    too_small_hits = contains_any(text, too_small_words)
    parts_hits = contains_any(text, parts_words)
    whole_hits = contains_any(text, whole_words)
    missing_hits = contains_any(text, missing_words)
    understood_hits = contains_any(text, understood_words)
    evidence_hits = contains_any(text, evidence_words)
    uncertainty_hits = contains_any(text, uncertainty_words)
    uncertainty_characterization_hits = contains_any(text, uncertainty_characterization_words)
    decision_hits = contains_any(text, decision_words)
    tradeoff_hits = contains_any(text, tradeoff_words)
    fragility_hits = contains_any(text, fragility_words)
    launch_hits = contains_any(text, launch_words)
    recurrence_hits_strong = contains_any(text, recurrence_words_strong)
    recurrence_hits_generic = contains_any(text, recurrence_words_generic)
    builder_hits = contains_any(text, builder_words)
    possibility_hits = contains_any(text, possibility_words)
    convergence_hits = contains_any(text, convergence_words)
    anti_convergence_hits = contains_any(text, anti_convergence_words)
    negated_closure_hits = contains_any(text, negated_closure_words)

    if negated_closure_hits:
        negated_tokens = {"decide", "recommend", "choose", "make a call"}
        decision_hits = [hit for hit in decision_hits if hit not in negated_tokens]
        anti_convergence_hits = sorted(set(anti_convergence_hits + negated_closure_hits))

    structural_signals: List[str] = []

    # expansion-when-defined
    if expansion_hits and define_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED)
        matches["expansion_when_defined"] = sorted(set(expansion_hits + define_hits))

    # concrete-form-too-small / abstraction overflow
    if concrete_hits and too_small_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL)
        matches["concrete_form_too_small"] = sorted(set(concrete_hits + too_small_hits))

    # parts/whole mismatch (legacy-compatible signal name retained)
    if parts_hits and whole_hits and missing_hits and understood_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED)
        matches["parts_whole_mismatch"] = sorted(set(parts_hits + whole_hits + missing_hits + understood_hits))

    if parts_hits and whole_hits and missing_hits:
        matches.setdefault("parts_whole_mismatch", sorted(set(parts_hits + whole_hits + missing_hits)))

    if evidence_hits or uncertainty_hits or uncertainty_characterization_hits:
        matches["uncertainty_evidence_demand"] = sorted(
            set(evidence_hits + uncertainty_hits + uncertainty_characterization_hits)
        )
    if uncertainty_characterization_hits:
        matches["uncertainty_characterization"] = sorted(set(uncertainty_characterization_hits))
    if decision_hits or tradeoff_hits:
        matches["decision_tradeoff_commitment"] = sorted(set(decision_hits + tradeoff_hits))
    if fragility_hits or launch_hits:
        matches["fragility_launch_trust"] = sorted(set(fragility_hits + launch_hits))
    if recurrence_hits_strong or builder_hits:
        matches["recurrence_systemization_strong"] = sorted(set(recurrence_hits_strong + builder_hits))
    if recurrence_hits_generic:
        matches["recurrence_pattern_generic"] = sorted(set(recurrence_hits_generic))
    if recurrence_hits_strong or recurrence_hits_generic or builder_hits:
        matches["recurrence_systemization"] = sorted(
            set(recurrence_hits_strong + recurrence_hits_generic + builder_hits)
        )
    if possibility_hits or convergence_hits or anti_convergence_hits:
        matches["open_possibility_space"] = sorted(set(possibility_hits + convergence_hits + anti_convergence_hits))
    if anti_convergence_hits:
        matches["anti_convergence_preference"] = sorted(set(anti_convergence_hits))
    if negated_closure_hits:
        matches["negated_closure_preference"] = sorted(set(negated_closure_hits))

    return RoutingFeatures(
        structural_signals=structural_signals,
        decision_pressure=score_from_matches(decision_hits, tradeoff_hits),
        evidence_demand=score_from_matches(evidence_hits, uncertainty_hits, uncertainty_characterization_hits),
        fragility_pressure=score_from_matches(fragility_hits, launch_hits),
        recurrence_potential=min(10, (2 * len(recurrence_hits_strong)) + (2 * len(builder_hits))),
        possibility_space_need=score_from_matches(possibility_hits, convergence_hits, anti_convergence_hits),
        detected_markers=matches,
    )


def explain_feature_matches(features: RoutingFeatures) -> dict[str, list[str]]:
    return dict(features.detected_markers)


def extract_structural_signals(task: str) -> List[str]:
    return extract_routing_features(task).structural_signals

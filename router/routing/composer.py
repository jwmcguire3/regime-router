from __future__ import annotations

from typing import List, Optional, Set, Tuple

from router.models import (
    CANONICAL_FAILURE_IF_OVERUSED,
    CANONICAL_DOMINANTS,
    FunctionType,
    LIBRARY,
    LinePrimitive,
    Regime,
    Stage,
)


class RegimeComposer:
    def compose(self, stage: Stage, risk_profile: Optional[Set[str]] = None, handoff_expected: bool = False) -> Regime:
        risk_profile = risk_profile or set()
        dominant = self._choose_dominant(stage, risk_profile)
        suppression = self._choose_suppressions(stage, risk_profile)
        shapes = self._choose_shapes(stage, risk_profile)
        tail = self._choose_tail(stage, handoff_expected, risk_profile)

        candidate_lines = [dominant] + suppression + shapes + ([tail] if tail else [])
        final_lines, rejected, reasons = self._resolve_conflicts(candidate_lines)
        final_lines = self._dedupe_and_trim(final_lines, max_lines=5)

        dom = next(l for l in final_lines if l.function == FunctionType.DOMINANCE)
        sup = [l for l in final_lines if l.function == FunctionType.SUPPRESSION]
        shp = [l for l in final_lines if l.function == FunctionType.SHAPE]
        tail_final = next((l for l in final_lines if l.function in (FunctionType.GATE, FunctionType.TRANSFER)), None)

        return Regime(
            name=f"{stage.value.title()} Core",
            stage=stage,
            dominant_line=dom,
            suppression_lines=sup,
            shape_lines=shp,
            tail_line=tail_final,
            rejected_lines=rejected,
            rejection_reasons=reasons,
            likely_failure_if_overused=CANONICAL_FAILURE_IF_OVERUSED[stage],
        )

    def _choose_dominant(self, stage: Stage, risk_profile: Set[str]) -> LinePrimitive:
        if stage == Stage.SYNTHESIS:
            return LIBRARY["SYN-D2"] if "sprawl" in risk_profile else LIBRARY["SYN-D1"]
        if stage == Stage.EPISTEMIC:
            return LIBRARY["EPI-D2"] if "elegant_theory_drift" in risk_profile else LIBRARY["EPI-D1"]
        return LIBRARY[CANONICAL_DOMINANTS[stage][0]]

    def _choose_suppressions(self, stage: Stage, risk_profile: Set[str]) -> List[LinePrimitive]:
        if stage == Stage.SYNTHESIS:
            chosen = ["SYN-P1"]
            if self._requires_synthesis_break_condition_pressure(risk_profile):
                chosen.append("SYN-P2")
        elif stage == Stage.EPISTEMIC:
            chosen = ["EPI-P1", "EPI-P2"]
        elif stage == Stage.ADVERSARIAL:
            chosen = ["ADV-P1", "ADV-P2"]
        elif stage == Stage.OPERATOR:
            chosen = ["OPR-P1"]
        elif stage == Stage.EXPLORATION:
            chosen = ["EXP-P1"]
        else:
            chosen = ["BLD-P1"]
        return [LIBRARY[i] for i in chosen]

    def _choose_shapes(self, stage: Stage, risk_profile: Set[str]) -> List[LinePrimitive]:
        if stage == Stage.EXPLORATION:
            chosen = ["EXP-S1"]
            if "need_reframing" in risk_profile:
                chosen.append("EXP-S2")
        elif stage == Stage.SYNTHESIS:
            chosen = [] if "high_stakes" in risk_profile else ["SYN-S1"]
        elif stage == Stage.EPISTEMIC:
            chosen = ["EPI-S1"]
        elif stage == Stage.ADVERSARIAL:
            chosen = ["ADV-S2"]
            if "single_point_failure" not in risk_profile:
                chosen.append("ADV-S1")
        elif stage == Stage.OPERATOR:
            chosen = ["OPR-S1", "OPR-S2"]
        else:
            chosen = ["BLD-S1", "BLD-S2"]
        return [LIBRARY[i] for i in chosen]

    @staticmethod
    def _requires_synthesis_break_condition_pressure(risk_profile: Set[str]) -> bool:
        high_risk_synthesis_conditions = {
            "coherence_over_truth",
            "false_unification",
            "high_stakes",
            "abstract_structural_task",
        }
        return bool(high_risk_synthesis_conditions & risk_profile)

    def _choose_tail(self, stage: Stage, handoff_expected: bool, risk_profile: Set[str]) -> Optional[LinePrimitive]:
        if stage == Stage.EXPLORATION and handoff_expected:
            return LIBRARY["EXP-T1"]
        if stage == Stage.EPISTEMIC and "strict" in risk_profile:
            return LIBRARY["EPI-G1"]
        if stage == Stage.ADVERSARIAL and handoff_expected:
            return LIBRARY["ADV-T1"]
        if stage == Stage.OPERATOR and "optionality" in risk_profile:
            return LIBRARY["OPR-G1"]
        if stage == Stage.BUILDER and handoff_expected:
            return LIBRARY["BLD-T1"]
        return None

    def _resolve_conflicts(self, lines: List[LinePrimitive]) -> Tuple[List[LinePrimitive], List[str], List[str]]:
        kept: List[LinePrimitive] = []
        rejected: List[str] = []
        reasons: List[str] = []

        for line in lines:
            conflict = False
            for existing in kept:
                if line.id in existing.incompatible_with or existing.id in line.incompatible_with:
                    rejected.append(line.id)
                    reasons.append(f"Rejected {line.id} because it conflicts with {existing.id}.")
                    conflict = True
                    break
            if not conflict:
                kept.append(line)

        dominance_lines = [l for l in kept if l.function == FunctionType.DOMINANCE]
        if len(dominance_lines) > 2:
            for extra in dominance_lines[2:]:
                kept = [l for l in kept if l.id != extra.id]
                rejected.append(extra.id)
                reasons.append(f"Rejected {extra.id} because more than two dominance lines weakens regime asymmetry.")

        if len(dominance_lines) == 2:
            a, b = dominance_lines[0], dominance_lines[1]
            if a.id in b.incompatible_with or b.id in a.incompatible_with:
                kept = [l for l in kept if l.id != b.id]
                rejected.append(b.id)
                reasons.append(f"Rejected {b.id} because it creates opposing motion with {a.id}.")

        return kept, rejected, reasons

    def _dedupe_and_trim(self, lines: List[LinePrimitive], max_lines: int = 5) -> List[LinePrimitive]:
        seen: Set[str] = set()
        deduped: List[LinePrimitive] = []
        for line in lines:
            if line.id not in seen:
                deduped.append(line)
                seen.add(line.id)

        priority = {
            FunctionType.DOMINANCE: 0,
            FunctionType.SUPPRESSION: 1,
            FunctionType.SHAPE: 2,
            FunctionType.GATE: 3,
            FunctionType.TRANSFER: 3,
        }
        deduped.sort(key=lambda x: priority[x.function])
        return deduped[:max_lines]

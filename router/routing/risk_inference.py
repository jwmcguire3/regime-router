from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional, Set

from router.models import STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED


def _load_feature_extraction_module():
    spec = spec_from_file_location("router.routing.feature_extraction", Path(__file__).with_name("feature_extraction.py"))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load feature extraction helpers from router/routing/feature_extraction.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_feature_extraction_module = _load_feature_extraction_module()


def infer_risk_profile(task: str, risk_profile: Optional[Set[str]]) -> Set[str]:
    inferred = set(risk_profile or set())
    text = task.lower()
    features = _feature_extraction_module.extract_routing_features(task)
    signals = set(features.structural_signals)

    if signals:
        inferred.add("abstract_structural_task")
    if (
        STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED in signals
        and any(k in text for k in ("single frame", "one frame", "unif", "compress", "organizing idea"))
    ):
        inferred.add("false_unification")
    if features.fragility_pressure >= 2:
        inferred.add("fragility_pressure")
    if features.evidence_demand >= 2:
        inferred.add("evidence_gap")
    if features.decision_pressure >= 2:
        inferred.add("decision_urgency")

    return inferred

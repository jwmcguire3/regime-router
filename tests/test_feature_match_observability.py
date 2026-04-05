from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.routing import explain_feature_matches, extract_routing_features


def test_explain_feature_matches_returns_plain_dict_without_changing_feature_computation() -> None:
    task = "Stress test this launch plan and list failure modes before deployment."

    features = extract_routing_features(task)
    markers_before = dict(features.detected_markers)
    explained = explain_feature_matches(features)

    assert isinstance(explained, dict)
    assert explained == markers_before

    features_again = extract_routing_features(task)
    assert features_again.detected_markers == markers_before

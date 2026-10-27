from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from evaluation.low_confidence import build_low_confidence_audit


def test_low_confidence_audit_uses_frozen_threshold_and_counts_groups_by_strata():
    flagged, report = build_low_confidence_audit(
        [0.45, 0.5, 0.62, 0.9],
        ["g1", "g1", "g2", "g3"],
        ["family-a", "family-a", "family-b", "family-b"],
        ["site-1", "site-1", "site-2", "site-2"],
        threshold=0.5,
        margin=0.05,
    )

    assert flagged.tolist() == [True, True, False, False]
    assert report["partition"] == "test"
    assert report["threshold_source"] == "validation"
    assert report["record_count"] == 4
    assert report["independent_group_count"] == 3
    assert report["low_confidence_record_count"] == 2
    assert report["low_confidence_independent_group_count"] == 1
    assert report["by_family"][0]["low_confidence_record_fraction"] == 1.0
    schema = json.loads(
        Path("evaluation/schema/low-confidence-audit.v1.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    ("scores", "groups", "margin", "message"),
    [
        ([0.2, np.nan], ["g1", "g2"], 0.1, "finite"),
        ([0.2, 0.8], ["g1"], 0.1, "align"),
        ([0.2, 0.8], ["g1", "g2"], 0.5, "0.5"),
    ],
)
def test_low_confidence_audit_rejects_invalid_contract(scores, groups, margin, message):
    with pytest.raises(ValueError, match=message):
        build_low_confidence_audit(
            scores,
            groups,
            ["family-a"] * len(scores),
            ["site-a"] * len(scores),
            threshold=0.5,
            margin=margin,
        )

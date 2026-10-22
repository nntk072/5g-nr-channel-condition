"""Regression tests for the binary LOS/NLOS label convention in spatial plots."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from integrations.sionna.los_oracle import SegmentVisibilityState

ROOT = Path(__file__).resolve().parents[2]
AUDITOR = runpy.run_path(str(ROOT / "scripts" / "audit_rt_spatial_holdout_labels.py"))
compare_label_to_visibility = AUDITOR["compare_label_to_visibility"]


@pytest.mark.parametrize(
    ("label", "state", "matches"),
    [
        (0, SegmentVisibilityState.BLOCKED, True),
        (1, SegmentVisibilityState.VISIBLE, True),
        (0, SegmentVisibilityState.VISIBLE, False),
        (1, SegmentVisibilityState.BLOCKED, False),
        (0, SegmentVisibilityState.AMBIGUOUS, False),
    ],
)
def test_label_audit_uses_one_for_clear_los(label, state, matches):
    assert compare_label_to_visibility(label, state) is matches


@pytest.mark.parametrize("label", (2, -1, True))
def test_label_audit_rejects_nonbinary_or_boolean_labels(label):
    with pytest.raises(ValueError, match="integer 0 or 1"):
        compare_label_to_visibility(label, SegmentVisibilityState.VISIBLE)


def test_spatial_plot_maps_label_colors_to_generator_contract():
    pytest.importorskip("matplotlib")
    plotter = runpy.run_path(str(ROOT / "scripts" / "plot_rt_tx_spatial_holdout.py"))
    assert plotter["CLASS_NAMES"] == {0: "Geometric NLOS", 1: "Geometric LOS"}
    assert plotter["CLASS_COLORS"][0] == "#e27a32"
    assert plotter["CLASS_COLORS"][1] == "#168a83"

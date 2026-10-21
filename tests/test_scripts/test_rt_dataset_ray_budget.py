"""Unit tests for the per-link RT ray-budget audit summaries."""

from __future__ import annotations

import pytest
from scripts.audit_rt_dataset_ray_budget import relative_l2_error, summarize_link_rows


def test_relative_l2_error_uses_reference_energy():
    assert relative_l2_error([3 + 0j, 4 + 0j], [0 + 0j, 4 + 0j]) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("value", "reference", "message"),
    [
        ([1 + 0j], [1 + 0j, 2 + 0j], "aligned complex arrays"),
        ([1.0], [1.0], "aligned complex arrays"),
        ([1 + 0j], [0 + 0j], "finite nonzero energy"),
    ],
)
def test_relative_l2_error_rejects_invalid_inputs(value, reference, message):
    with pytest.raises(ValueError, match=message):
        relative_l2_error(value, reference)


def test_summary_reports_path_changes_and_errors_by_label():
    rows = [
        {
            "sample_index": 0,
            "label": 0,
            "samples_per_src": 100,
            "valid_path_count": 2,
            "relative_cfr_l2_to_reference": 0.2,
        },
        {
            "sample_index": 1,
            "label": 1,
            "samples_per_src": 100,
            "valid_path_count": 3,
            "relative_cfr_l2_to_reference": 0.1,
        },
        {
            "sample_index": 0,
            "label": 0,
            "samples_per_src": 1000,
            "valid_path_count": 4,
            "relative_cfr_l2_to_reference": 0.0,
        },
        {
            "sample_index": 1,
            "label": 1,
            "samples_per_src": 1000,
            "valid_path_count": 3,
            "relative_cfr_l2_to_reference": 0.0,
        },
    ]

    summary = summarize_link_rows(rows, [100, 1000])

    assert summary["reference_samples_per_src"] == 1000
    assert summary["by_budget"]["100"]["path_count_changes_vs_reference"] == 1
    assert summary["by_budget"]["100"]["max_relative_cfr_l2"] == pytest.approx(0.2)
    assert summary["by_budget"]["100"]["by_label"]["1"]["p95_relative_cfr_l2"] == pytest.approx(0.1)


def test_summary_requires_each_sample_at_reference_budget():
    rows = [
        {
            "sample_index": 0,
            "label": 0,
            "samples_per_src": 100,
            "valid_path_count": 1,
            "relative_cfr_l2_to_reference": 0.2,
        }
    ]
    with pytest.raises(ValueError, match="reference-budget row"):
        summarize_link_rows(rows, [100, 1000])

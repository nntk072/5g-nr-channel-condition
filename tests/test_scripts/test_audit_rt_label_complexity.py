import numpy as np
import pytest
from datasets.channel import ChannelDataset
from scripts.audit_rt_label_complexity import summarize_rt_dataset


def _make_rt_dataset(path):
    metadata = [
        {
            "valid_paths_full": 2,
            "valid_paths_direct_oracle": 1,
            "los_plus_indirect_paths": True,
            "oracle_direct_power": 1.0,
            "residual_indirect_power": 0.25,
        },
        {
            "valid_paths_full": 1,
            "valid_paths_direct_oracle": 1,
            "los_plus_indirect_paths": False,
            "oracle_direct_power": 1.0,
            "residual_indirect_power": 0.0,
        },
        {
            "valid_paths_full": 3,
            "valid_paths_direct_oracle": 0,
            "los_plus_indirect_paths": False,
            "oracle_direct_power": 0.0,
            "residual_indirect_power": 0.4,
        },
        {
            "valid_paths_full": 0,
            "valid_paths_direct_oracle": 0,
            "los_plus_indirect_paths": False,
            "oracle_direct_power": 0.0,
            "residual_indirect_power": 0.0,
        },
    ]
    report = {
        "schema": "loslab.raytraced_generation_report.v1",
        "requested_per_class": 2,
        "attempt_counts": {
            "full_solver_calls": 5,
            "full_solver_empty_paths": 1,
            "attempts_started": 5,
        },
        "full_solver_outcomes": {"empty_paths": 1, "accepted": 4},
        "accepted_counts": {"geometric_nlos": 2, "geometric_los": 2, "total": 4},
        "selection_note": "Empty full-solver path outcomes are counted, not exported.",
    }
    dataset = ChannelDataset(
        h=np.ones((4, 1, 8), dtype=np.complex64),
        y=np.asarray([1, 1, 0, 0], dtype=np.uint8),
        group=np.asarray(["g0", "g1", "g2", "g3"]),
        domain=np.asarray(["rt_test"] * 4),
        source=np.asarray(["sionna_rt"] * 4),
        metadata=tuple(metadata),
        generation_report=report,
        label_semantics="geometric_direct_path_oracle",
    )
    dataset.save(path)


def test_rt_label_complexity_reports_mechanisms_and_selection(tmp_path):
    path = tmp_path / "rt.npz"
    _make_rt_dataset(path)

    report = summarize_rt_dataset(path)

    assert report["accepted_mechanisms"] == {
        "los_direct_only": 1,
        "los_mixed": 1,
        "nlos_indirect_paths": 1,
        "nlos_no_path": 1,
    }
    assert report["direct_to_indirect_power_ratio_db_for_los_mixed"]["median"] == pytest.approx(
        10 * np.log10(4)
    )
    assert report["generation_selection"]["empty_path_candidate_fraction"] == pytest.approx(0.2)
    assert report["generation_selection"]["empty_path_class_attribution"] == (
        "not recorded by legacy generation report"
    )


def test_rt_label_complexity_rejects_label_oracle_disagreement(tmp_path):
    path = tmp_path / "rt.npz"
    _make_rt_dataset(path)
    with np.load(path, allow_pickle=False) as archive:
        y = archive["y"].copy()
        y[0] = 0
    dataset = ChannelDataset.load(path)
    ChannelDataset(
        h=dataset.h,
        y=y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=dataset.metadata,
        generation_report=dataset.generation_report,
        label_semantics=dataset.label_semantics,
    ).save(path)

    with pytest.raises(ValueError, match="geometry label and direct-path oracle disagree"):
        summarize_rt_dataset(path)

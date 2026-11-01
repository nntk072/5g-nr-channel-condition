import hashlib
from pathlib import Path

import numpy as np
import pytest
from artifacts.provenance import canonical_hash
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


def test_rt_label_complexity_verifies_v4_execution_provenance(tmp_path):
    path = tmp_path / "rt.npz"
    _make_rt_dataset(path)
    dataset = ChannelDataset.load(path)
    generation = dict(dataset.generation_report)
    generation.update(
        {
            "schema": "loslab.raytraced_generation_report.v4",
            "status": "completed",
            "seed": 42,
            "rt_run_identity": {"complete": True, "cache_key_sha256": "a" * 64},
        }
    )
    execution = {
        "schema": "loslab.rt_execution_provenance.v1",
        "schema_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "datasets/schema/rt-execution-provenance.v1.schema.json").read_bytes()
        ).hexdigest(),
        "started_at_utc": "2026-11-01T00:00:00.000000Z",
        "completed_at_utc": "2026-11-01T00:00:01.000000Z",
        "python_executable": "/usr/bin/python3",
        "command_argv": ["loslab-generate-rt", "--seed", "42"],
        "base_seed": 42,
        "attempt_seed_scheme": "base_seed_plus_zero_based_candidate_attempt_index",
        "candidate_attempt_count": 5,
        "terminal_status": "completed",
        "failure_stage": None,
        "rt_cache_key_sha256": "a" * 64,
    }
    generation["execution_provenance"] = execution
    generation["execution_provenance_sha256"] = canonical_hash(execution)
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=dataset.metadata,
        generation_report=generation,
        label_semantics=dataset.label_semantics,
    ).save(path)
    summarize_rt_dataset(path)

    tampered = dict(execution)
    tampered["command_argv"] = ["different-command"]
    generation["execution_provenance"] = tampered
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=dataset.metadata,
        generation_report=generation,
        label_semantics=dataset.label_semantics,
    ).save(path)
    with pytest.raises(ValueError, match="execution provenance digest is invalid"):
        summarize_rt_dataset(path)

    tampered["base_seed"] = 43
    generation["execution_provenance_sha256"] = canonical_hash(tampered)
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=dataset.metadata,
        generation_report=generation,
        label_semantics=dataset.label_semantics,
    ).save(path)
    with pytest.raises(ValueError, match="does not match generation inputs or outcomes"):
        summarize_rt_dataset(path)

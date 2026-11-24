"""Standardized split lineage: run identity, sample uniqueness and report bindings."""

from __future__ import annotations

from dataclasses import replace

import pytest
from datasets.split_lineage import (
    derive_standardized_run_split_subjects,
    make_standardized_run_split_registry,
)
from datasets.split_policy import SplitPolicy
from tests.test_datasets.test_split_policy_v2 import _standardized_run


def test_run_roster_must_be_nonempty_and_typed() -> None:
    with pytest.raises(ValueError, match="at least one standardized run"):
        derive_standardized_run_split_subjects([])
    with pytest.raises(TypeError, match="StandardizedChannelRun values"):
        derive_standardized_run_split_subjects(["not-a-run"])


def test_run_ids_must_be_unique() -> None:
    first = _standardized_run("same-run", seed=1, draw_count=1)
    second = _standardized_run("same-run", seed=2, draw_count=1)
    with pytest.raises(ValueError, match="run IDs must be unique"):
        derive_standardized_run_split_subjects([first, second])


def test_sample_ids_must_be_unique_across_runs() -> None:
    first = _standardized_run("run-a", seed=1, draw_count=1)
    record = first.records[0]
    clash = replace(
        _standardized_run("run-b", seed=2, draw_count=1), records=(replace(record, run_id="run-b"),)
    )
    with pytest.raises(ValueError, match="sample IDs must be unique across runs"):
        derive_standardized_run_split_subjects([first, clash])


def test_report_binds_runs_subjects_and_partitions() -> None:
    run = _standardized_run("report-run", seed=5, draw_count=4)
    subjects, report = derive_standardized_run_split_subjects([run])
    assert report["run_count"] == 1
    assert report["draw_count"] == len(subjects) == 4
    assert sum(report["source_families"].values()) == 4
    assert len(report["subject_roster_sha256"]) == 64
    assert len(report["partition_roster_sha256"]) == 64


def test_registry_builder_reports_policy_and_split_counts() -> None:
    run = _standardized_run("registry-run", seed=9, draw_count=8)
    policy = SplitPolicy.release_v2(seed="registry-report")
    registry, report = make_standardized_run_split_registry([run], policy=policy)
    assert report["policy_sha256"] == policy.to_dict()["policy_sha256"]
    assert report["registry_sha256"] == registry.to_dict()["registry_sha256"]
    assert sum(report["parent_counts_by_split"].values()) == 8

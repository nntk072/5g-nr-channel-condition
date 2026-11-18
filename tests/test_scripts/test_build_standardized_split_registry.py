from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from datasets.split_policy import (
    SplitPolicy,
    read_split_policy_registry,
)
from datasets.standardized import (
    StandardizedChannelRecord,
    StandardizedChannelRun,
    StandardizedFamily,
    write_standardized_run,
)
from scripts.build_standardized_split_registry import (
    build_standardized_split_registry,
    main,
)


def _run(run_id: str, *, seed: int, draw_count: int = 8) -> StandardizedChannelRun:
    offsets = (np.arange(8) - 4) * 15_000.0
    h_path = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex64)
    template = StandardizedChannelRecord(
        sample_id=f"{run_id}-sample-0",
        run_id=run_id,
        sample_index=0,
        family=StandardizedFamily.TDL,
        profile_model="A",
        profile_los_component=False,
        h_path=h_path,
        tau_s=np.zeros((1, 1, 1), dtype=np.float64),
        cfr=np.broadcast_to(h_path[..., 0, None], (1, 1, 1, 1, 1, 8)).copy(),
        subcarrier_offsets_hz=offsets,
        metadata={
            "profile_model": "A",
            "profile_los_component": False,
            "delay_profile_semantics": "scalable_tr_38_901_profile",
            "delay_reference": "profile_excess_delay",
            "seed": seed,
            "spec_version": "19.2",
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 15_000.0,
            "num_subcarriers": 8,
            "sampled_bandwidth_hz": 120_000.0,
            "num_time_steps": 1,
            "rx_antenna_count": 1,
            "tx_antenna_count": 1,
            "rx_corr_mat_real": None,
            "rx_corr_mat_imag": None,
        },
    )
    return StandardizedChannelRun(
        run_id,
        tuple(
            replace(
                template,
                sample_id=f"{run_id}-sample-{index:04d}",
                sample_index=index,
            )
            for index in range(draw_count)
        ),
    )


def _write_policy(path, *, seed: str) -> None:
    path.write_text(
        json.dumps(SplitPolicy.release_v2(seed=seed).to_dict(), sort_keys=True),
        encoding="utf-8",
    )


def test_standardized_split_builder_reads_immutable_runs_and_publishes_registry(tmp_path, capsys):
    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("standardized-batch", seed=101))
    manifest = store / stored.relative_path
    policy = tmp_path / "policy.json"
    _write_policy(policy, seed="stored-standardized-builder")
    output = tmp_path / "release-candidate"

    assert (
        main(
            [
                "--run",
                f"batch-a={manifest}",
                "--policy",
                str(policy),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)

    saved_report = json.loads((output / "build-report.json").read_text(encoding="utf-8"))
    assert saved_report["status"] == "CANDIDATE_NOT_EXTERNALLY_REVIEWED"
    assert saved_report["qualification_status"] == "NOT_QUALIFIED"
    assert saved_report["inputs"][0]["manifest_sha256"]
    assert saved_report["inputs"][0]["record_sha256"]
    assert cli_result["registry_sha256"] == saved_report["registry_sha256"]
    assert cli_result["qualification_status"] == "NOT_QUALIFIED"
    registry = read_split_policy_registry(output / "split-registry.json")
    assert len(registry.subjects) == 8
    assert set(registry.report()["splits"]) == {"train", "validation", "calibration", "test"}
    assert (
        saved_report["registry_file_sha256"]
        == hashlib.sha256((output / "split-registry.json").read_bytes()).hexdigest()
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_standardized_split_registry(
            [("batch-a", manifest)], policy_path=policy, output_dir=output
        )


def test_standardized_split_builder_rejects_reused_seed_index_without_outputs(tmp_path):
    store = tmp_path / "store"
    first = write_standardized_run(store, _run("run-a", seed=202, draw_count=1))
    second = write_standardized_run(store, _run("run-b", seed=202, draw_count=1))
    policy = tmp_path / "policy.json"
    _write_policy(policy, seed="reused-standardized-index")
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="partition_id values must uniquely identify"):
        build_standardized_split_registry(
            [("first", store / first.relative_path), ("second", store / second.relative_path)],
            policy_path=policy,
            output_dir=output,
        )

    assert not output.exists()

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets.split_policy import (
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitPolicyRegistry,
    SplitSubject,
    VerifiedOpenSSHArtifact,
    bind_split_children,
    make_split_policy_registry,
)
from scripts import evaluate_rt_nr_trajectory_forecasts as forecast_script


def _signed_registry() -> SplitPolicyRegistry:
    parents = tuple(
        SplitSubject(
            subject_id="forecast-parent" if index == 0 else f"forecast-parent-{index}",
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="synthetic-test",
        )
        for index in range(64)
    )
    registry = make_split_policy_registry(
        parents, policy=SplitPolicy.release_v2(seed="signed-forecast-test")
    )
    return bind_split_children(
        registry,
        (
            SplitChildBinding("sample-a", parents[0].subject_id, "nr_trajectory_sample"),
            SplitChildBinding("sample-b", parents[0].subject_id, "nr_trajectory_sample"),
            SplitChildBinding("other-child", parents[0].subject_id, "nr_record"),
        ),
    )


def _patch_forecast_inputs(
    monkeypatch: pytest.MonkeyPatch,
    sample_ids: tuple[str, ...],
    *,
    record_parent_subject_id: str = "forecast-parent",
) -> None:
    monkeypatch.setattr(
        forecast_script,
        "_load_stored_trajectory",
        lambda _store, trajectory_id: SimpleNamespace(
            content_sha256="a" * 64,
            trajectory_id=trajectory_id,
        ),
    )
    monkeypatch.setattr(
        forecast_script,
        "read_nr_trajectory",
        lambda _store, _stored: SimpleNamespace(
            sample_ids=sample_ids,
            records=tuple(
                SimpleNamespace(
                    provenance={
                        "split_subject_id": record_parent_subject_id,
                        "split_child_id": sample_id,
                        "split_child_kind": (
                            "nr_record" if sample_id == "other-child" else "nr_trajectory_sample"
                        ),
                    }
                )
                for sample_id in sample_ids
            ),
        ),
    )
    monkeypatch.setattr(
        forecast_script,
        "evaluate_trajectory_forecasts",
        lambda _trajectory, *, split, split_registry: [
            {
                "split": split,
                "trajectory_id": "trajectory-a",
                "forecast_target_s": 1.0,
                "method": "persistence",
            }
        ],
    )
    monkeypatch.setattr(forecast_script, "summarize_forecast_rows", lambda _rows: {})
    monkeypatch.setattr(forecast_script, "versions", lambda: {})
    monkeypatch.setattr(forecast_script, "package_source_sha256", lambda: "b" * 64)


def test_signed_forecast_uses_registry_child_roster_and_exact_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _signed_registry()
    verified = VerifiedOpenSSHArtifact(
        payload=b"exact verified registry bytes",
        artifact_sha256="c" * 64,
        signature_sha256="d" * 64,
        allowed_signers_sha256="e" * 64,
        signer_id="research@example.org",
        namespace="umcd-split-policy-v2",
    )
    monkeypatch.setattr(
        forecast_script,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (registry, verified),
    )
    _patch_forecast_inputs(monkeypatch, ("sample-a", "sample-b"))
    spec_path = tmp_path / "trajectory.json"
    spec_path.write_text(
        json.dumps({"trajectory_id": "trajectory-a", "scene": "scene-a"}),
        encoding="utf-8",
    )

    report = forecast_script.evaluate_frozen_trajectories(
        tmp_path / "store",
        tmp_path / "registry.json",
        [spec_path],
        split_signature_path=tmp_path / "registry.sig",
        trusted_split_allowed_signers_path=tmp_path / "allowed_signers",
        split_signer_id="research@example.org",
    )

    assert report["split_lock_sha256"] == verified.artifact_sha256
    assert report["split_provenance"] == {
        "mode": "signed_split_policy_registry",
        "signature_sha256": verified.signature_sha256,
        "trusted_allowed_signers_sha256": verified.allowed_signers_sha256,
        "signer_id": verified.signer_id,
        "signature_namespace": verified.namespace,
        "registry_policy_sha256": registry.policy.to_dict()["policy_sha256"],
        "registry_schema": registry.to_dict()["schema"],
        "trajectory_child_kind": "nr_trajectory_sample",
    }
    assert set(report["metrics_by_split"]) == set(registry.policy.split_fractions)


def test_signed_forecast_requires_complete_signature_configuration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires signature"):
        forecast_script.evaluate_frozen_trajectories(
            tmp_path / "store",
            tmp_path / "registry.json",
            [tmp_path / "trajectory.json"],
            split_signature_path=tmp_path / "registry.sig",
        )


def test_signed_forecast_rejects_child_of_wrong_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _signed_registry()
    verified = VerifiedOpenSSHArtifact(
        payload=b"verified",
        artifact_sha256="c" * 64,
        signature_sha256="d" * 64,
        allowed_signers_sha256="e" * 64,
        signer_id="research@example.org",
        namespace="umcd-split-policy-v2",
    )
    monkeypatch.setattr(
        forecast_script,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (registry, verified),
    )
    _patch_forecast_inputs(monkeypatch, ("sample-a", "other-child"))
    spec_path = tmp_path / "trajectory.json"
    spec_path.write_text(
        json.dumps({"trajectory_id": "trajectory-a", "scene": "scene-a"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong signed child kind"):
        forecast_script.evaluate_frozen_trajectories(
            tmp_path / "store",
            tmp_path / "registry.json",
            [spec_path],
            split_signature_path=tmp_path / "registry.sig",
            trusted_split_allowed_signers_path=tmp_path / "allowed_signers",
            split_signer_id="research@example.org",
        )


def test_signed_forecast_rejects_record_parent_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _signed_registry()
    verified = VerifiedOpenSSHArtifact(
        payload=b"verified",
        artifact_sha256="c" * 64,
        signature_sha256="d" * 64,
        allowed_signers_sha256="e" * 64,
        signer_id="research@example.org",
        namespace="umcd-split-policy-v2",
    )
    monkeypatch.setattr(
        forecast_script,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (registry, verified),
    )
    _patch_forecast_inputs(
        monkeypatch,
        ("sample-a", "sample-b"),
        record_parent_subject_id="different-parent",
    )
    spec_path = tmp_path / "trajectory.json"
    spec_path.write_text(
        json.dumps({"trajectory_id": "trajectory-a", "scene": "scene-a"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split_subject_id differs"):
        forecast_script.evaluate_frozen_trajectories(
            tmp_path / "store",
            tmp_path / "registry.json",
            [spec_path],
            split_signature_path=tmp_path / "registry.sig",
            trusted_split_allowed_signers_path=tmp_path / "allowed_signers",
            split_signer_id="research@example.org",
        )

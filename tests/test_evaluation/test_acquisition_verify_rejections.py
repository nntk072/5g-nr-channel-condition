"""Rejection contracts of the acquisition-package consistency verifier.

Each case edits one declared relation in an otherwise consistent synthetic package and
checks the verifier's named finding. The fixtures are the same synthetic records used by
the verifier's main test module; none of them is measurement evidence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from evaluation.acquisition import verify as verify_module
from evaluation.acquisition.verify import verify_acquisition_package
from tests.test_evaluation.test_acquisition_package_verifier import (
    _error_text,
    _make_package,
    _put,
)

V02 = "g3.03-acquisition-package.v0.2"
V04 = "g3.03-acquisition-package.v0.4"
Mutation = Callable[[Path, dict[str, Any]], None]


def _load(tmp_path: Path, schema: str = V02) -> tuple[Path, dict[str, Any]]:
    package_path = _make_package(tmp_path, schema_version=schema)
    return package_path, json.loads(package_path.read_text(encoding="utf-8"))


def _save(package_path: Path, package: dict[str, Any]) -> None:
    package_path.write_text(json.dumps(package), encoding="utf-8")


def _file_document(root: Path, package: dict[str, Any], file_id: str) -> dict[str, Any]:
    entry = next(item for item in package["files"] if item["file_id"] == file_id)
    return json.loads((root / entry["path"]).read_bytes())


def _rewrite_file(root: Path, package: dict[str, Any], file_id: str, document: Any) -> None:
    _put(root, package, file_id, json.dumps(document).encode("utf-8"))


def _rewrite_scorer(mutate: Callable[[dict[str, Any]], None]) -> Mutation:
    def apply(root: Path, package: dict[str, Any]) -> None:
        document = _file_document(root, package, "file.scorer.input")
        mutate(document)
        _rewrite_file(root, package, "file.scorer.input", document)

    return apply


def _rewrite_output(mutate: Callable[[dict[str, Any]], None]) -> Mutation:
    def apply(root: Path, package: dict[str, Any]) -> None:
        document = _file_document(root, package, "file.evaluator.output")
        mutate(document)
        _rewrite_file(root, package, "file.evaluator.output", document)

    return apply


def _package_edit(mutate: Callable[[dict[str, Any]], None]) -> Mutation:
    def apply(root: Path, package: dict[str, Any]) -> None:
        mutate(package)

    return apply


CASES = [
    pytest.param(
        _package_edit(lambda d: d.update(proposal_status="accepted")),
        "package must use a supported proposal schema and remain proposal_only",
        id="not-proposal-only",
    ),
    pytest.param(
        _package_edit(lambda d: d.update(schema_version="g3.03-acquisition-package.v9")),
        "unsupported acquisition package schema version",
        id="unsupported-schema-version",
    ),
    pytest.param(
        _package_edit(lambda d: d["evaluator_binding"].update(execution_status="executed")),
        "evaluator binding must remain not_executed_proposal_only",
        id="binding-executed",
    ),
    pytest.param(
        _package_edit(lambda d: d["campaigns"][0].update(study_id="study.other")),
        "study_id differs from package study_id",
        id="campaign-other-study",
    ),
    pytest.param(
        _package_edit(lambda d: d["campaigns"][0].update(start_utc="2099-01-01T00:00:00Z")),
        "interval starts after it ends",
        id="campaign-inverted-interval",
    ),
    pytest.param(
        _package_edit(lambda d: d["protocols"][0].update(split_manifest_file_id="file.absent")),
        "references unknown split manifest file",
        id="protocol-unknown-split-file",
    ),
    pytest.param(
        _package_edit(
            lambda d: d["protocols"][0].update(split_manifest_file_id="file.scorer.input")
        ),
        "split-manifest file binding differs",
        id="protocol-split-binding-differs",
    ),
    pytest.param(
        _package_edit(lambda d: d["protocols"][0].update(split_manifest_canonical_sha256="0" * 64)),
        "canonical split hash differs",
        id="protocol-split-hash-differs",
    ),
    pytest.param(
        _package_edit(lambda d: d["uncertainty_records"][0].update(sample_ids=["sample.absent"])),
        "references unknown sample",
        id="uncertainty-unknown-sample",
    ),
    pytest.param(
        _package_edit(
            lambda d: d["external_attestations"][0].update(signature_file_id="file.absent")
        ),
        "references unknown signature file",
        id="attestation-unknown-signature-file",
    ),
    pytest.param(
        _package_edit(
            lambda d: d["evaluator_binding"]["scorer_input"].update(file_id="file.absent")
        ),
        "evaluator_binding.scorer_input references unknown file_id",
        id="binding-unknown-file",
    ),
    pytest.param(
        _package_edit(lambda d: d["evaluator_binding"]["scorer_input"].update(exact_bytes_size=1)),
        "exact-byte reference disagrees with file bytes",
        id="binding-exact-bytes-disagree",
    ),
    pytest.param(
        _package_edit(
            lambda d: next(f for f in d["files"] if f["file_id"] == "file.scorer.input").update(
                role="other"
            )
        ),
        "evaluator_binding.scorer_input file must have inventory role",
        id="binding-wrong-inventory-role",
    ),
    pytest.param(
        _rewrite_scorer(lambda d: d.update(schema="loslab.other.v1")),
        "scorer input schema does not match evaluator input contract",
        id="scorer-schema-mismatch",
    ),
    pytest.param(
        _rewrite_scorer(lambda d: d.update(full_split_manifest="not-an-object")),
        "scorer input full_split_manifest must be an object",
        id="scorer-manifest-not-object",
    ),
    pytest.param(
        _rewrite_scorer(lambda d: d.update(expected_strata=[])),
        "expected_strata differs from",
        id="scorer-expected-strata-differs",
    ),
    pytest.param(
        _rewrite_output(lambda d: d.update(schema="loslab.other.v1")),
        "evaluator output schema does not match output contract",
        id="output-schema-mismatch",
    ),
    pytest.param(
        _rewrite_output(lambda d: d.update(strata="not-a-list")),
        "evaluator output strata must be an array",
        id="output-strata-not-array",
    ),
    pytest.param(
        _rewrite_output(lambda d: d.update(strata=[1])),
        "evaluator output strata contain non-object entries",
        id="output-stratum-not-object",
    ),
    pytest.param(
        _rewrite_output(lambda d: d["strata"].append(dict(d["strata"][0]))),
        "evaluator output strata contain non-objects or duplicate keys",
        id="output-duplicate-stratum",
    ),
    pytest.param(
        _rewrite_output(lambda d: d["strata"][0].update(material_id=7)),
        "evaluator output stratum keys must be strings",
        id="output-stratum-key-not-string",
    ),
    pytest.param(
        _rewrite_output(lambda d: d["strata"][0].update(material_id="material.other")),
        "evaluator output strata do not exactly match scorer input expected_strata",
        id="output-stratum-mismatch",
    ),
]


@pytest.fixture
def no_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the declared JSON Schema so the semantic cross-checks run on their own."""
    import jsonschema

    monkeypatch.setattr(
        jsonschema.Draft202012Validator, "iter_errors", lambda self, instance: iter(())
    )


SEMANTIC_CASES = [
    pytest.param(
        _package_edit(lambda d: d["protocols"][0].update(target_frequency_hz=1.0)),
        "target frequency differs from scorer input",
        id="protocol-frequency-differs",
    ),
    pytest.param(
        _package_edit(lambda d: d["evaluator_binding"].pop("split_manifest")),
        "evaluator_binding.split_manifest is missing",
        id="binding-missing-reference",
    ),
    pytest.param(
        _package_edit(lambda d: d["evaluator_binding"]["scorer_input"].update(file_id=5)),
        "evaluator_binding.scorer_input.file_id must be a string",
        id="binding-file-id-not-string",
    ),
]


@pytest.mark.usefixtures("no_json_schema")
@pytest.mark.parametrize(("mutate", "message"), SEMANTIC_CASES)
def test_verifier_semantic_checks_reject_without_schema_help(
    tmp_path: Path, mutate: Mutation, message: str
) -> None:
    package_path, package = _load(tmp_path)
    mutate(tmp_path, package)
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert message in _error_text(report)


@pytest.mark.parametrize(("mutate", "message"), CASES)
def test_verifier_reports_each_declared_relation_violation(
    tmp_path: Path, mutate: Mutation, message: str
) -> None:
    package_path, package = _load(tmp_path)
    mutate(tmp_path, package)
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert message in _error_text(report)


def test_verifier_rejects_non_object_evaluator_output(tmp_path: Path) -> None:
    package_path, package = _load(tmp_path)
    _put(tmp_path, package, "file.evaluator.output", b"[]\n")
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "evaluator output must be a JSON object" in _error_text(report)


def test_verifier_rejects_unparseable_bound_artifact(tmp_path: Path) -> None:
    package_path, package = _load(tmp_path)
    _put(tmp_path, package, "file.evaluator.output", b"not json")
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert _error_text(report)


def test_verifier_rejects_package_root_that_is_not_a_directory(tmp_path: Path) -> None:
    package_path, _ = _load(tmp_path)
    report = verify_acquisition_package(package_path, package_root=package_path)

    assert report["status"] == "inconsistent"
    assert "package root must be a directory" in _error_text(report)


def test_verifier_bounds_inventory_file_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path, _ = _load(tmp_path)
    monkeypatch.setattr(verify_module, "MAX_INVENTORY_FILES", 1)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "inventory exceeds 1 files" in _error_text(report)


def test_verifier_bounds_total_inventory_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path, package = _load(tmp_path)
    # Budget exactly the bytes of every inventory file except the last one, so the
    # final read finds no remaining budget.
    earlier_bytes = sum(item["size_bytes"] for item in package["files"][:-1])
    monkeypatch.setattr(verify_module, "MAX_TOTAL_FILE_BYTES", earlier_bytes)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "inventory exceeds total byte-read bound" in _error_text(report)


def test_verifier_reports_platform_without_directory_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path, _ = _load(tmp_path)
    monkeypatch.setattr(os, "supports_dir_fd", set())

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "cannot securely open package root" in _error_text(report)


def _set_angular(update: Callable[[dict[str, Any]], None]) -> Mutation:
    def apply(root: Path, package: dict[str, Any]) -> None:
        update(package["angular_measurements"][0])

    return apply


def _set_csi(update: Callable[[dict[str, Any]], None]) -> Mutation:
    def apply(root: Path, package: dict[str, Any]) -> None:
        update(package["csi_measurements"][0])

    return apply


def _zero_power(record: dict[str, Any]) -> None:
    record["power_per_bin_w"] = [[0.0 for _ in row] for row in record["power_per_bin_w"]]


def _negative_first_power(record: dict[str, Any]) -> None:
    record["power_per_bin_w"][0][0] = -1.0


def _scale_solid_angles(record: dict[str, Any]) -> None:
    record["solid_angle_sr"] = [[2.0 * value for value in row] for row in record["solid_angle_sr"]]


SEMANTIC_BREADTH_CASES = [
    pytest.param(
        _package_edit(lambda d: [item.update(size_bytes="12") for item in d["files"]]),
        "size_bytes must be an integer",
        id="size-not-integer",
    ),
    pytest.param(
        _package_edit(
            lambda d: d["attestations"]["calibration_truth"].update(
                attestation_id=d["external_attestations"][0]["attestation_id"]
            )
        ),
        "is not globally unique",
        id="legacy-attestation-id-collision",
    ),
    pytest.param(
        _package_edit(
            lambda d: d["attestations"]["calibration_truth"].update(
                evidence_file_ids=["file.absent"]
            )
        ),
        "legacy attestation references unknown file_id",
        id="legacy-attestation-unknown-file",
    ),
    pytest.param(
        _package_edit(
            lambda d: d["evaluator_binding"].update(run_configuration_file_ids="not-a-list")
        ),
        "evaluator_binding.run_configuration_file_ids must be an array",
        id="binding-refs-not-array",
    ),
    pytest.param(
        _set_csi(lambda r: r.update(shape=[0, 1, 1, 1])),
        "shape must contain four positive integers",
        id="csi-shape-not-positive",
    ),
    pytest.param(
        _set_csi(lambda r: r["h_floor"].update(method="other-method")),
        "h_floor method must be",
        id="csi-h-floor-method",
    ),
    pytest.param(
        _set_angular(lambda r: r.update(phi_edges_deg=list(reversed(r["phi_edges_deg"])))),
        "edges must be strictly increasing",
        id="angular-edges-not-increasing",
    ),
    pytest.param(
        _set_angular(lambda r: r.update(missing_bin_indices=[[999, 999]])),
        "invalid missing-bin index",
        id="angular-missing-bin-out-of-range",
    ),
    pytest.param(
        _set_angular(lambda r: r.update(missing_bin_indices=[[0, 0], [0, 0]])),
        "repeats a missing-bin index",
        id="angular-missing-bin-repeated",
    ),
    pytest.param(
        _set_angular(_negative_first_power),
        "must be finite and nonnegative",
        id="angular-negative-power",
    ),
    pytest.param(
        _set_angular(_zero_power),
        "must have a positive sum",
        id="angular-zero-power",
    ),
    pytest.param(
        _set_angular(_scale_solid_angles),
        "must match the spherical bin geometry",
        id="angular-solid-angle-geometry",
    ),
]


@pytest.mark.usefixtures("no_json_schema")
@pytest.mark.parametrize(("mutate", "message"), SEMANTIC_BREADTH_CASES)
def test_verifier_breadth_checks_reject_without_schema_help(
    tmp_path: Path, mutate: Mutation, message: str
) -> None:
    package_path, package = _load(tmp_path, V04)
    mutate(tmp_path, package)
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert message in _error_text(report)


def _noise_entry(package: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in package["files"] if item["role"] == "noise_measurement")


def _csi_with_noise(package: dict[str, Any]) -> dict[str, Any]:
    return package["csi_measurements"][0]


NOISE_CASES = [
    pytest.param(
        lambda root, d: _noise_entry(d).update(role="other"),
        "h_floor noise input",
        id="noise-wrong-role",
    ),
    pytest.param(
        lambda root, d: _noise_entry(d).update(media_type="application/octet-stream"),
        "must declare media_type 'application/x-npy'",
        id="noise-wrong-media-type",
    ),
    pytest.param(
        lambda root, d: _noise_entry(d).update(sample_ids=["sample.absent"]),
        "h_floor noise input",
        id="noise-bound-to-other-sample",
    ),
    pytest.param(
        lambda root, d: _csi_with_noise(d)["h_floor"].update(noise_input_file_ids=["file.absent"]),
        "h_floor",
        id="noise-input-absent",
    ),
    pytest.param(
        lambda root, d: (root / _noise_entry(d)["path"]).unlink(),
        "has no captured, hash-verified bytes",
        id="noise-bytes-unavailable",
    ),
    pytest.param(
        lambda root, d: _noise_entry(d).update(path="noise/not-numpy.bin"),
        "must use a .npy inventory path",
        id="noise-not-npy-path",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), NOISE_CASES)
def test_verifier_rejects_unbound_or_unreadable_noise_inputs(
    tmp_path: Path, mutate: Callable[[Path, dict[str, Any]], None], message: str
) -> None:
    package_path, package = _load(tmp_path, V04)
    mutate(tmp_path, package)
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert message in _error_text(report)


def test_verifier_rejects_duplicate_legacy_attestation_ids(tmp_path: Path) -> None:
    package_path, package = _load(tmp_path)
    package["attestations"]["physical_independence"]["attestation_id"] = package["attestations"][
        "calibration_truth"
    ]["attestation_id"]
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "is not globally unique" in _error_text(report)


def test_verifier_rejects_unknown_run_configuration_reference(tmp_path: Path) -> None:
    package_path, package = _load(tmp_path)
    package["evaluator_binding"]["run_configuration_file_ids"] = ["file.absent"]
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "run_configuration_file_ids references unknown file_id" in _error_text(report)


def test_verifier_rejects_fit_covariance_reference_to_unknown_file(tmp_path: Path) -> None:
    package_path, package = _load(tmp_path)
    package["fits"][0]["parameters"][0]["covariance_file_id"] = "file.absent"
    _save(package_path, package)

    report = verify_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "references unknown covariance file" in _error_text(report)

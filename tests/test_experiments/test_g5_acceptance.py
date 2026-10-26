from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from experiments.g5_acceptance import (
    G5_SNR_CURVE_BOOTSTRAP_DRAWS,
    G5_SNR_CURVE_BOOTSTRAP_SEED,
    G5_SNR_CURVE_INFERENCE_METHOD,
    G5_SNR_CURVE_INFERENCE_UNIT,
    G5_SNR_CURVE_PAIRED_CONTRAST,
    evaluate_g5_snr_curve_acceptance,
    freeze_g5_snr_acceptance_criteria,
    load_g5_snr_acceptance_contract,
    validate_g5_preregistration_order,
)


def _criteria() -> dict[str, object]:
    return {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0, 5.0],
        "channel_seeds": list(range(100, 120)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 4,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [0.1, 0.05],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.02, 0.01],
    }


def _timestamped_criteria() -> dict[str, object]:
    criteria = _criteria()
    criteria["schema"] = "umcd.g5_ordered_impairments_snr_acceptance.v2"
    criteria["preregistered_utc"] = "2026-10-25T09:00:00Z"
    return criteria


def _freeze(criteria: dict[str, object] | None = None) -> dict[str, object]:
    result = freeze_g5_snr_acceptance_criteria(
        _criteria() if criteria is None else criteria,
        snr_values=(0.0, 5.0),
        channel_seeds=tuple(range(100, 120)),
        channel_estimator="sionna_default_ls",
        blocks_per_channel=4,
    )
    assert result is not None
    return result


def _inference(
    *,
    corrected: tuple[list[float], list[float]] = ([0.02, 0.06], [0.01, 0.04]),
    improvement: tuple[list[float], list[float]] = ([0.03, 0.2], [0.02, 0.1]),
    cluster_count: int = 20,
) -> dict[str, object]:
    snr_values = (0.0, 5.0)
    clean = ([0.01, 0.03], [0.0, 0.02])
    uncorrected = ([0.05, 0.25], [0.03, 0.14])
    return {
        "status": "available",
        "cluster_count": cluster_count,
        "channel_seeds": list(range(100, 100 + cluster_count)),
        "unit": G5_SNR_CURVE_INFERENCE_UNIT,
        "bootstrap_seed": G5_SNR_CURVE_BOOTSTRAP_SEED,
        "bootstrap_draws": G5_SNR_CURVE_BOOTSTRAP_DRAWS,
        "family_size": len(snr_values) * 4,
        "method": G5_SNR_CURVE_INFERENCE_METHOD,
        "paired_contrast_definition": G5_SNR_CURVE_PAIRED_CONTRAST,
        "simultaneous_radius": 0.05,
        "by_condition": {
            condition: [
                {"snr_db": snr, "simultaneous_95_percent_interval": interval}
                for snr, interval in zip(snr_values, intervals, strict=True)
            ]
            for condition, intervals in (
                ("clean", clean),
                ("uncorrected", uncorrected),
                ("corrected", corrected),
            )
        },
        "by_paired_contrast": [
            {"snr_db": snr, "simultaneous_95_percent_interval": interval}
            for snr, interval in zip(snr_values, improvement, strict=True)
        ],
    }


def test_g5_acceptance_freezes_contract_without_receiver_dependency() -> None:
    criteria = _criteria()
    frozen = _freeze(criteria)

    assert frozen == criteria
    assert _freeze(frozen) == frozen


def test_g5_timestamped_acceptance_contract_freezes_declared_preregistration_time() -> None:
    criteria = _timestamped_criteria()
    frozen = freeze_g5_snr_acceptance_criteria(
        criteria,
        snr_values=(0.0, 5.0),
        channel_seeds=tuple(range(100, 120)),
        channel_estimator="sionna_default_ls",
        blocks_per_channel=4,
    )

    assert frozen == criteria
    assert validate_g5_preregistration_order(
        frozen, receiver_run_started_utc="2026-10-26T09:00:00Z"
    ) == {
        "declared_temporal_order_status": "preregistration_precedes_receiver_run",
        "preregistered_utc": "2026-10-25T09:00:00Z",
        "receiver_run_started_utc": "2026-10-26T09:00:00Z",
        "timestamp_authority": "not_established_by_this_verifier",
    }


@pytest.mark.parametrize(
    "receiver_run_started_utc",
    ("2026-10-25T09:00:00Z", "2026-10-24T09:00:00Z", "not-a-time"),
)
def test_g5_timestamped_acceptance_rejects_nonprior_or_invalid_run_time(
    receiver_run_started_utc: str,
) -> None:
    with pytest.raises(ValueError, match="must precede|valid ISO-8601|ending in Z"):
        validate_g5_preregistration_order(
            _timestamped_criteria(), receiver_run_started_utc=receiver_run_started_utc
        )


@pytest.mark.parametrize("preregistered_utc", ("2026-10-25T09:00:00+00:00", "bad"))
def test_g5_timestamped_acceptance_requires_canonical_utc_timestamp(
    preregistered_utc: str,
) -> None:
    criteria = _timestamped_criteria()
    criteria["preregistered_utc"] = preregistered_utc

    with pytest.raises(ValueError, match="ending in Z|valid ISO-8601"):
        _freeze(criteria)


def test_timestamped_acceptance_v2_schema_requires_declared_utc_time() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-acceptance.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    criteria = _timestamped_criteria()

    jsonschema.Draft202012Validator(schema).validate(criteria)
    del criteria["preregistered_utc"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(criteria)


def test_signed_v2_contract_admission_binds_timestamp_before_receiver_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    criteria = _timestamped_criteria()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(criteria, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    signature_path = tmp_path / "contract.sig"
    signature_path.write_bytes(b"signature fixture")
    allowed_signers_path = tmp_path / "allowed_signers"
    allowed_signers_path.write_bytes(b"allowed signers fixture")

    with pytest.raises(ValueError, match="v2 requires a verified signature"):
        load_g5_snr_acceptance_contract(contract_path)

    observed: dict[str, object] = {}

    def verify(
        payload: bytes,
        verified_signature_path: Path,
        verified_allowed_signers_path: Path,
        *,
        signer_id: str,
        namespace: str,
    ) -> dict[str, str]:
        observed.update(
            payload=payload,
            signature_path=verified_signature_path,
            allowed_signers_path=verified_allowed_signers_path,
            signer_id=signer_id,
            namespace=namespace,
        )
        return {
            "signer_id": signer_id,
            "namespace": namespace,
            "signature_sha256": "1" * 64,
            "allowed_signers_sha256": "2" * 64,
        }

    monkeypatch.setattr("experiments.g5_acceptance.verify_openssh_signature_for_bytes", verify)
    loaded, identity = load_g5_snr_acceptance_contract(
        contract_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers_path,
        signer_id="reviewer@example.org",
    )

    assert loaded == criteria
    assert identity is not None
    assert identity["preregistered_utc"] == criteria["preregistered_utc"]
    assert isinstance(observed["payload"], bytes)
    assert identity["contract_payload_sha256"] == hashlib.sha256(observed["payload"]).hexdigest()
    assert observed["namespace"] == "umcd-g5-snr-acceptance-v1"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_g5_contract_loader_rejects_file_aliases(tmp_path: Path, alias_kind: str) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(_criteria(), sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    alias_path = tmp_path / f"{alias_kind}-contract.json"
    if alias_kind == "symlink":
        alias_path.symlink_to(contract_path)
        expected = "Too many levels of symbolic links|single-link regular file"
    else:
        alias_path.hardlink_to(contract_path)
        expected = "single-link regular file"

    with pytest.raises(ValueError, match=expected):
        load_g5_snr_acceptance_contract(alias_path)


def test_timestamped_acceptance_contract_verifies_real_openssh_signature(
    tmp_path: Path,
) -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required for signature verification")
    key_path = tmp_path / "acceptance_signer"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True,
        capture_output=True,
    )
    public_key = subprocess.run(
        [ssh_keygen, "-y", "-f", str(key_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    allowed_signers_path = tmp_path / "allowed_signers"
    allowed_signers_path.write_text(
        f'reviewer@example.org namespaces="umcd-g5-snr-acceptance-v1" {public_key}\n',
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(_timestamped_criteria(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-f",
            str(key_path),
            "-n",
            "umcd-g5-snr-acceptance-v1",
            str(contract_path),
        ],
        check=True,
        capture_output=True,
    )
    signature_path = Path(f"{contract_path}.sig")

    loaded, identity = load_g5_snr_acceptance_contract(
        contract_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers_path,
        signer_id="reviewer@example.org",
    )

    assert loaded == _timestamped_criteria()
    assert identity is not None
    assert identity["signer_id"] == "reviewer@example.org"
    assert identity["namespace"] == "umcd-g5-snr-acceptance-v1"
    assert identity["preregistered_utc"] == "2026-10-25T09:00:00Z"

    altered = _timestamped_criteria()
    altered["preregistered_utc"] = "2026-10-26T09:00:00Z"
    contract_path.write_text(
        json.dumps(altered, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="signature is invalid"):
        load_g5_snr_acceptance_contract(
            contract_path,
            signature_path=signature_path,
            trusted_allowed_signers_path=allowed_signers_path,
            signer_id="reviewer@example.org",
        )


def test_reviewer_policy_requires_distinct_signed_custody_and_authorizes_reviewer(
    tmp_path: Path,
) -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required for signature verification")

    def create_key(name: str, principal: str, namespace: str) -> tuple[Path, Path]:
        private = tmp_path / name
        subprocess.run(
            [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
            check=True,
            capture_output=True,
        )
        public = subprocess.run(
            [ssh_keygen, "-y", "-f", str(private)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        allowed = tmp_path / f"{name}.allowed_signers"
        allowed.write_text(f'{principal} namespaces="{namespace}" {public}\n', encoding="utf-8")
        return private, allowed

    acceptance_private, acceptance_keyring = create_key(
        "acceptance", "reviewer@example.org", "umcd-g5-snr-acceptance-v1"
    )
    policy_private, policy_keyring = create_key(
        "policy", "custodian@example.net", "umcd-g5-snr-reviewer-policy-v1"
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(_timestamped_criteria(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-f",
            str(acceptance_private),
            "-n",
            "umcd-g5-snr-acceptance-v1",
            str(contract_path),
        ],
        check=True,
        capture_output=True,
    )
    policy_path = tmp_path / "reviewer-policy.json"
    policy = {
        "schema": "umcd.g5_snr_reviewer_policy.v1",
        "policy_id": "test-reviewer-policy",
        "signers": [
            {
                "principal": "reviewer@example.org",
                "organization": "Independent Test Lab",
                "roles": ["independent_reviewer"],
                "allowed_purposes": ["g5_ordered_impairments_snr_preregistration"],
            }
        ],
    }
    policy_path.write_text(
        json.dumps(policy, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-f",
            str(policy_private),
            "-n",
            "umcd-g5-snr-reviewer-policy-v1",
            str(policy_path),
        ],
        check=True,
        capture_output=True,
    )

    _, identity = load_g5_snr_acceptance_contract(
        contract_path,
        signature_path=Path(f"{contract_path}.sig"),
        trusted_allowed_signers_path=acceptance_keyring,
        signer_id="reviewer@example.org",
        reviewer_policy_path=policy_path,
        reviewer_policy_signature_path=Path(f"{policy_path}.sig"),
        trusted_reviewer_allowed_signers_path=policy_keyring,
        reviewer_policy_signer_id="custodian@example.net",
    )
    assert identity is not None
    assert identity["reviewer_policy"]["reviewer_principal"] == "reviewer@example.org"
    assert identity["reviewer_policy"]["custodian_principal"] == "custodian@example.net"

    shared_keyring = tmp_path / "shared.allowed_signers"
    shared_keyring.write_bytes(acceptance_keyring.read_bytes() + policy_keyring.read_bytes())
    with pytest.raises(ValueError, match="keyring must differ"):
        load_g5_snr_acceptance_contract(
            contract_path,
            signature_path=Path(f"{contract_path}.sig"),
            trusted_allowed_signers_path=shared_keyring,
            signer_id="reviewer@example.org",
            reviewer_policy_path=policy_path,
            reviewer_policy_signature_path=Path(f"{policy_path}.sig"),
            trusted_reviewer_allowed_signers_path=shared_keyring,
            reviewer_policy_signer_id="custodian@example.net",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("snr_db", [0.0], "SNR grid"),
        ("channel_seeds", list(range(101, 121)), "channel seeds"),
        ("blocks_per_channel", 3, "block count"),
        ("minimum_independent_channels", 19, "at least 20"),
        ("maximum_corrected_macro_bler", [0.1], "cover every SNR"),
        ("minimum_uncorrected_minus_corrected_macro_bler", [0.02, float("nan")], "canonical JSON"),
        ("maximum_corrected_macro_bler", [10**400, 0.05], "finite real values"),
    ),
)
def test_g5_acceptance_rejects_contract_drift_and_nonfinite_values(
    field: str, value: object, message: str
) -> None:
    criteria = deepcopy(_criteria())
    criteria[field] = value

    with pytest.raises(ValueError, match=message):
        _freeze(criteria)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("snr_db", [float(index) for index in range(65)], "bounded limit of 64 points"),
        (
            "channel_seeds",
            list(range(100, 100 + 8_193)),
            "bounded limit of 16384 transport blocks",
        ),
    ),
)
def test_g5_acceptance_bounds_direct_contracts_before_snapshot(
    field: str, value: object, message: str
) -> None:
    criteria = deepcopy(_criteria())
    criteria[field] = value

    with pytest.raises(ValueError, match=message):
        _freeze(criteria)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("snr_db", [0.0, 0.0], "strictly increasing"),
        ("channel_seeds", [100] * 20, "unique nonnegative integers"),
        ("maximum_corrected_macro_bler", [1.5, 0.05], "in \\[0, 1\\]"),
        ("minimum_uncorrected_minus_corrected_macro_bler", [-0.1, 0.01], "in \\[0, 1\\]"),
    ),
)
def test_g5_acceptance_scorer_revalidates_contract_before_deciding(
    field: str, value: object, message: str
) -> None:
    criteria = deepcopy(_criteria())
    criteria[field] = value

    with pytest.raises(ValueError, match=message):
        evaluate_g5_snr_curve_acceptance(criteria, curve_inference=_inference())


def test_g5_acceptance_requires_pointwise_simultaneous_bounds_to_pass() -> None:
    result = evaluate_g5_snr_curve_acceptance(_freeze(), curve_inference=_inference())

    assert result["status"] == "PASS"
    assert result["pass"] is True
    assert len(result["point_decisions"]) == 2


def test_g5_acceptance_fails_when_either_bound_misses_at_one_snr() -> None:
    result = evaluate_g5_snr_curve_acceptance(
        _freeze(),
        curve_inference=_inference(
            corrected=([0.02, 0.06], [0.01, 0.051]),
            improvement=([0.03, 0.2], [0.009, 0.1]),
        ),
    )

    assert result["status"] == "FAIL"
    assert result["pass"] is False
    assert result["point_decisions"][1]["pass"] is False


def test_g5_acceptance_stays_not_evaluable_below_frozen_cluster_floor() -> None:
    result = evaluate_g5_snr_curve_acceptance(
        _freeze(), curve_inference=_inference(cluster_count=19)
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert result["pass"] is False
    assert result["point_decisions"] == []


def test_g5_acceptance_requires_inference_to_cover_exact_frozen_seed_roster() -> None:
    criteria = _criteria()
    criteria["channel_seeds"] = list(range(100, 121))

    with pytest.raises(ValueError, match="cluster count does not match the frozen seed roster"):
        evaluate_g5_snr_curve_acceptance(criteria, curve_inference=_inference())


def test_g5_acceptance_rejects_same_count_with_different_frozen_seed_ids() -> None:
    inference = _inference()
    inference["channel_seeds"] = list(range(200, 220))

    with pytest.raises(ValueError, match="channel seeds do not match"):
        evaluate_g5_snr_curve_acceptance(_freeze(), curve_inference=inference)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (lambda inference: inference.__setitem__("family_size", 2), "frozen G5 method"),
        (lambda inference: inference["by_condition"].pop("clean"), "complete frozen condition"),
        (lambda inference: inference.__setitem__("simultaneous_radius", float("nan")), "radius"),
    ),
)
def test_g5_acceptance_rejects_inference_metadata_drift(change, message: str) -> None:
    inference = _inference()
    change(inference)

    with pytest.raises(ValueError, match=message):
        evaluate_g5_snr_curve_acceptance(_freeze(), curve_inference=inference)


def test_g5_acceptance_does_not_treat_boolean_snr_as_numeric_one() -> None:
    criteria = _criteria()
    criteria["snr_db"] = [1.0]
    criteria["maximum_corrected_macro_bler"] = [0.1]
    criteria["minimum_uncorrected_minus_corrected_macro_bler"] = [0.02]
    inference = _inference()
    inference["family_size"] = 4
    inference["by_condition"] = {
        condition: [dict(rows[0], snr_db=True)]
        for condition, rows in inference["by_condition"].items()
    }
    inference["by_paired_contrast"] = [dict(inference["by_paired_contrast"][0], snr_db=True)]

    with pytest.raises(ValueError, match="SNR ordering differs"):
        evaluate_g5_snr_curve_acceptance(criteria, curve_inference=inference)


@pytest.mark.parametrize(
    "interval",
    (
        [float("-inf"), 0.1],
        [10**400, 0.1],
        [0.2, 0.1],
        [-0.1, 0.1],
        [False, 0.1],
        [0.0, 1.1],
    ),
)
def test_g5_acceptance_rejects_malformed_interval_endpoints(interval: list[object]) -> None:
    inference = _inference()
    inference["by_condition"]["corrected"][0]["simultaneous_95_percent_interval"] = interval

    with pytest.raises(ValueError, match="finite numbers|reversed or out of domain"):
        evaluate_g5_snr_curve_acceptance(_freeze(), curve_inference=inference)

"""End-to-end CPU PUSCH checks for G5 ordered multipath impairments."""

import hashlib
import io
import json
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

import experiments.audit_g5_ordered_impairments_result as audit_module  # noqa: E402
import experiments.g5_ordered_impairments_receiver as experiment  # noqa: E402
from experiments.audit_g5_ordered_impairments_result import (  # noqa: E402
    _recompute_waveform_diagnostics,
    audit_result_file,
)
from experiments.g5_ordered_impairments_receiver import (  # noqa: E402
    build_hash_bound_result,
    run_independent_ordered_impairment_probe,
    run_independent_ordered_impairment_snr_curve,
    run_ordered_impairment_probe,
    write_hash_bound_result,
)


def _fake_snr_point(**kwargs: Any) -> dict[str, Any]:
    seeds = kwargs["channel_seeds"]
    blocks = kwargs["blocks_per_channel"]
    variance = kwargs["noise_variance"]
    records = [
        {
            "channel_seed": seed,
            "payload_seed": seed + 1_000_000,
            "noise_seed": seed + 1_200_000,
            "tx_phase_noise_seed": seed + 1_300_000,
            "rx_phase_noise_seed": seed + 1_400_000,
            "path_delays_samples": [0, 2],
            "path_coefficients_real_imag": [[0.9, -0.1], [0.3, 0.2]],
            "crc_pass_by_condition": {
                condition: [True] * blocks for condition in ("clean", "uncorrected", "corrected")
            },
            "transport_blocks": [
                {
                    "transport_block_id": f"channel-{seed}:payload-{seed + 1_000_000}:block-{index}",
                    "block_index": index,
                    "payload_seed": seed + 1_000_000,
                    "crc_pass_by_condition": {
                        condition: True for condition in ("clean", "uncorrected", "corrected")
                    },
                }
                for index in range(blocks)
            ],
        }
        for seed in seeds
    ]
    return {
        "schema": (
            "umcd.g5_ordered_impairments_independent_channels.v21"
            if kwargs.get("device", "cpu") != "cpu"
            else "umcd.g5_ordered_impairments_independent_channels.v15"
        ),
        "config": {
            "channel_seeds": list(seeds),
            "blocks_per_channel": blocks,
            "payload_seed_offset": 1_000_000,
            "noise_seed_offset": 1_200_000,
            "tx_phase_noise_seed_offset": 1_300_000,
            "rx_phase_noise_seed_offset": 1_400_000,
            "channel_estimator": {"name": kwargs["channel_estimator"]},
            "waveform_awgn_complex_variance": variance,
            "nominal_snr_db": -10.0 * np.log10(variance),
            "mean_transmit_waveform_power": 1.0,
            "transmit_waveform_power_by_channel": [1.0] * len(seeds),
            "noisy_receiver_input_sha256_by_channel": ["0" * 64 for _ in seeds],
            **(
                {"execution_device": kwargs["device"]}
                if kwargs.get("device", "cpu") != "cpu"
                else {}
            ),
            **(
                {"receiver_configuration": {"device": kwargs["device"]}}
                if kwargs.get("device", "cpu") != "cpu"
                else {}
            ),
            "per_channel_draws": records,
        },
        "paired_outcomes": {
            condition: {
                "blocks": len(seeds) * blocks,
                "crc_failures": 0,
                "bler": 0.0,
                "channel_macro_bler": 0.0,
            }
            for condition in ("clean", "uncorrected", "corrected")
        },
        "acceptance": {},
    }


def test_bounded_waveform_buffer_caps_high_water_mark() -> None:
    output = experiment._BoundedBytesIO(4)
    assert output.write(b"abcd") == 4
    output.seek(0)
    assert output.write(b"wxyz") == 4
    with pytest.raises(ValueError, match="compressed waveform evidence"):
        output.write(b"!")


def test_independent_snr_curve_pairs_seed_roster_and_binds_variance(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_probe(**kwargs):
        calls.append(kwargs)
        return _fake_snr_point(**kwargs)

    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", fake_probe)
    report = experiment.run_independent_ordered_impairment_snr_curve(
        snr_db=(0.0, 10.0), channel_seeds=(17, 23), blocks_per_channel=2
    )

    assert [call["channel_seeds"] for call in calls] == [(17, 23), (17, 23)]
    assert [call["blocks_per_channel"] for call in calls] == [2, 2]
    assert [call["noise_variance"] for call in calls] == pytest.approx([1.0, 0.1])
    assert [point["snr_db"] for point in report["points"]] == [0.0, 10.0]
    assert report["config"]["transport_blocks_across_curve"] == 8
    assert report["curve_inference"]["status"] == "available"
    assert report["schema"] == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9"
    assert report["curve_inference"]["family_size"] == 8
    assert len(report["curve_inference"]["by_condition"]["clean"]) == 2
    assert len(report["curve_inference"]["by_paired_contrast"]) == 2
    assert all(
        "mean_channel_macro_bler_improvement" in row
        for row in report["curve_inference"]["by_paired_contrast"]
    )
    assert report["acceptance"]["pass"] is False
    assert report["acceptance"]["criteria_frozen_before_run"]["status"] == "descriptive_only"


def test_independent_snr_curve_routes_explicit_cuda_device_and_versions_report(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_probe(**kwargs):
        calls.append(kwargs)
        return _fake_snr_point(**kwargs)

    monkeypatch.setattr(experiment, "_validate_execution_device", lambda device: device)
    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", fake_probe)
    report = experiment.run_independent_ordered_impairment_snr_curve(
        snr_db=(0.0, 10.0),
        channel_seeds=(17, 23),
        blocks_per_channel=2,
        device="cuda:0",
    )

    assert [call["device"] for call in calls] == ["cuda:0", "cuda:0"]
    assert report["schema"] == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v12"
    assert report["config"]["execution_device"] == "cuda:0"
    assert all(
        point["experiment"]["schema"] == "umcd.g5_ordered_impairments_independent_channels.v21"
        and point["experiment"]["config"]["execution_device"] == "cuda:0"
        for point in report["points"]
    )
    result_path = tmp_path / "cuda-curve.json"
    write_hash_bound_result(result_path, build_hash_bound_result(report))
    audit = audit_result_file(result_path)
    assert audit["checks"]["execution_device_binding"] is True
    assert audit["checks"]["paired_roster_noise_and_crc_summaries"] is True


def test_independent_snr_curve_applies_pre_run_family_acceptance_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        experiment,
        "run_independent_ordered_impairment_probe",
        _fake_snr_point,
    )
    criteria = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0, 10.0],
        "channel_seeds": list(range(17, 37)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 2,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [1.0, 1.0],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.0, 0.0],
    }
    report = experiment.run_independent_ordered_impairment_snr_curve(
        snr_db=(0.0, 10.0),
        channel_seeds=tuple(range(17, 37)),
        blocks_per_channel=2,
        acceptance_criteria=criteria,
    )

    assert report["schema"] == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7"
    assert report["curve_inference"]["channel_seeds"] == list(range(17, 37))
    assert report["acceptance"]["status"] == "PASS"
    assert report["acceptance"]["pass"] is True
    assert len(report["acceptance"]["point_decisions"]) == 2
    assert (
        report["acceptance"]["criteria_frozen_before_run"]["status"]
        == "contract_supplied_before_run"
    )
    result = build_hash_bound_result(report, created_utc="2026-10-26T00:00:00+00:00")
    result_path = tmp_path / "accepted-curve.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    audit = audit_result_file(result_path)
    assert audit["checks"]["acceptance_contract_decision"] is True

    tampered_report = json.loads(json.dumps(report))
    tampered_report["curve_inference"]["channel_seeds"] = list(
        reversed(tampered_report["curve_inference"]["channel_seeds"])
    )
    tampered_result = build_hash_bound_result(
        tampered_report, created_utc="2026-10-26T00:00:00+00:00"
    )
    tampered_path = tmp_path / "accepted-curve-wrong-seed-roster.json"
    tampered_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    tampered_audit = audit_result_file(tampered_path)
    assert tampered_audit["checks"]["channel_cluster_curve_inference"] is False


def test_independent_snr_curve_rejects_acceptance_contract_roster_mismatch_before_probe(
    monkeypatch,
) -> None:
    def unexpected_probe(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("receiver must not run for a mismatched acceptance contract")

    monkeypatch.setattr(
        experiment,
        "run_independent_ordered_impairment_probe",
        unexpected_probe,
    )
    criteria = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0],
        "channel_seeds": list(range(31, 51)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 1,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [1.0],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.0],
    }
    with pytest.raises(ValueError, match="channel seeds must exactly match"):
        experiment.run_independent_ordered_impairment_snr_curve(
            snr_db=(0.0,),
            channel_seeds=tuple(range(31, 50)) + (51,),
            blocks_per_channel=1,
            acceptance_criteria=criteria,
        )


def test_independent_snr_curve_rejects_contract_below_executable_cluster_floor(
    monkeypatch,
) -> None:
    def unexpected_probe(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("receiver must not run below the acceptance cluster floor")

    monkeypatch.setattr(
        experiment,
        "run_independent_ordered_impairment_probe",
        unexpected_probe,
    )
    criteria = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0],
        "channel_seeds": list(range(31, 51)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 1,
        "minimum_independent_channels": 19,
        "maximum_corrected_macro_bler": [1.0],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.0],
    }
    with pytest.raises(ValueError, match="at least 20"):
        experiment.run_independent_ordered_impairment_snr_curve(
            snr_db=(0.0,),
            channel_seeds=tuple(range(31, 51)),
            blocks_per_channel=1,
            acceptance_criteria=criteria,
        )


@pytest.mark.parametrize(
    ("interval", "branch", "message"),
    (
        ([float("-inf"), 0.1], "corrected", "finite numbers"),
        ([10**400, 0.1], "corrected", "finite numbers"),
        ([0.2, 0.1], "corrected", "reversed or out of domain"),
        ([-0.1, 0.1], "corrected", "reversed or out of domain"),
        ([False, 0.1], "corrected", "finite numbers"),
        ([0.0, 1.1], "improvement", "reversed or out of domain"),
        ([0.2, -0.2], "improvement", "reversed or out of domain"),
    ),
)
def test_snr_acceptance_rejects_malformed_simultaneous_intervals(
    interval: list[float], branch: str, message: str
) -> None:
    criteria = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0],
        "channel_seeds": list(range(17, 37)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 1,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [1.0],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.0],
    }
    frozen = experiment._freeze_independent_snr_acceptance_criteria(
        criteria,
        snr_values=(0.0,),
        channel_seeds=tuple(range(17, 37)),
        channel_estimator="sionna_default_ls",
        blocks_per_channel=1,
    )
    corrected_interval = [0.0, 1.0] if branch == "corrected" else interval
    improvement_interval = [0.0, 1.0] if branch == "improvement" else interval
    inference = {
        "status": "available",
        "cluster_count": 20,
        "by_condition": {
            "corrected": [{"snr_db": 0.0, "simultaneous_95_percent_interval": corrected_interval}]
        },
        "by_paired_contrast": [
            {"snr_db": 0.0, "simultaneous_95_percent_interval": improvement_interval}
        ],
    }

    with pytest.raises(ValueError, match=message):
        experiment._evaluate_independent_snr_curve_acceptance(frozen, curve_inference=inference)


def test_independent_snr_curve_retains_each_condition_input_across_snr(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_probe(**kwargs):
        calls.append(kwargs)
        report = _fake_snr_point(**kwargs)
        report["acceptance"] = {
            "criteria_frozen_before_run": {
                "max_samplewise_reference_absolute_error": 0.1,
                "min_wrong_order_rmse": 0.5,
            },
            "observed": {
                "max_samplewise_reference_absolute_error": 0.01,
                "min_wrong_order_rmse": 1.0,
            },
        }
        output = kwargs["waveform_evidence_output"]
        if output is not None:
            channel_count = len(kwargs["channel_seeds"])
            shape = (channel_count, 1, 1, 4)
            variance = np.float32(kwargs["noise_variance"])
            tx = np.ones(shape, dtype=np.complex64)
            payload = io.BytesIO()
            np.savez_compressed(
                payload,
                channel_seeds=np.asarray(kwargs["channel_seeds"], dtype=np.int64),
                tx_waveform=tx,
                tx_phase_noise_rad=np.zeros(shape, dtype=np.float32),
                rx_phase_noise_rad=np.zeros(shape, dtype=np.float32),
                actual_capture=tx,
                awgn_standard_normal_real=np.ones(shape, dtype=np.float32),
                awgn_standard_normal_imag=np.zeros(shape, dtype=np.float32),
                clean_receiver_input=tx,
                uncorrected_receiver_input=tx * (1.0 + variance),
                corrected_receiver_input=tx * (1.0 + variance / 2.0),
            )
            evidence_bytes = payload.getvalue()
            output.write_bytes(evidence_bytes)
            report["config"]["waveform_evidence"] = {
                "schema": "umcd.g5_ordered_impairments_waveform_evidence.v3",
                "file_name": output.name,
                "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                "byte_count": len(evidence_bytes),
                "channel_count": channel_count,
                "arrays": [
                    "actual_capture",
                    "awgn_standard_normal_imag",
                    "awgn_standard_normal_real",
                    "channel_seeds",
                    "clean_receiver_input",
                    "corrected_receiver_input",
                    "rx_phase_noise_rad",
                    "tx_phase_noise_rad",
                    "tx_waveform",
                    "uncorrected_receiver_input",
                ],
            }
            report["config"]["receiver_input_sha256_by_condition"] = {
                "clean": ["0" * 64] * channel_count,
                "uncorrected": ["1" * 64] * channel_count,
                "corrected": ["2" * 64] * channel_count,
            }
            report["schema"] = "umcd.g5_ordered_impairments_independent_channels.v19"
        return report

    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", fake_probe)
    evidence_path = tmp_path / "curve.waveforms.npz"
    report = experiment.run_independent_ordered_impairment_snr_curve(
        snr_db=(0.0, 10.0),
        channel_seeds=(17, 23),
        blocks_per_channel=2,
        waveform_evidence_output=evidence_path,
    )

    assert all(call["waveform_evidence_output"] is not None for call in calls)
    assert all(call["include_condition_inputs"] is True for call in calls)
    manifest = report["config"]["waveform_evidence"]
    assert manifest["file_name"] == evidence_path.name
    assert manifest["schema"] == "umcd.g5_ordered_impairments_waveform_evidence.v4"
    assert manifest["snr_db"] == [0.0, 10.0]
    with np.load(evidence_path, allow_pickle=False) as archive:
        assert "clean_receiver_input_by_snr" in archive.files
        assert archive["clean_receiver_input_by_snr"].shape == (2, 2, 1, 1, 4)
        assert np.array_equal(
            archive["uncorrected_receiver_input_by_snr"][0],
            np.ones((2, 1, 1, 4), dtype=np.complex64) * 2.0,
        )
        assert np.array_equal(
            archive["uncorrected_receiver_input_by_snr"][1],
            np.ones((2, 1, 1, 4), dtype=np.complex64) * 1.1,
        )
    assert all(
        point["experiment"]["config"]["waveform_evidence"] == manifest for point in report["points"]
    )
    assert all(point["experiment"]["schema"].endswith(".v19") for point in report["points"])
    assert list(tmp_path.iterdir()) == [evidence_path]

    monkeypatch.setattr(
        audit_module,
        "_recompute_waveform_diagnostics",
        lambda config, records, payload, *, snr_index=None: (0.01, 1.0),
    )
    result_path = tmp_path / "curve.json"
    result_path.write_text(
        json.dumps(build_hash_bound_result(report, created_utc="2026-10-26T00:00:00+00:00")),
        encoding="utf-8",
    )
    audit = audit_result_file(result_path)
    assert audit["checks"]["waveform_equation_recomputation"] is True


def test_snr_curve_waveform_destination_is_rejected_before_receiver(
    monkeypatch, tmp_path: Path
) -> None:
    existing = tmp_path / "already-there.npz"
    existing.write_bytes(b"preserve")

    def unexpected_probe(**kwargs):
        raise AssertionError(f"receiver invoked before evidence-path rejection: {kwargs}")

    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", unexpected_probe)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run_independent_ordered_impairment_snr_curve(
            snr_db=(0.0,), waveform_evidence_output=existing
        )
    assert existing.read_bytes() == b"preserve"


def test_independent_snr_curve_rejects_invalid_grid_and_total_work_before_receiver(
    monkeypatch,
) -> None:
    def unexpected_probe(**kwargs):
        raise AssertionError(f"receiver invoked before validation: {kwargs}")

    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", unexpected_probe)
    with pytest.raises(ValueError, match="strictly increasing"):
        experiment.run_independent_ordered_impairment_snr_curve(snr_db=(5.0, 5.0))
    with pytest.raises(ValueError, match="exceeding the bounded limit"):
        experiment.run_independent_ordered_impairment_snr_curve(
            snr_db=(0.0, 1.0),
            channel_seeds=tuple(range(1_025)),
            blocks_per_channel=8,
        )
    with pytest.raises(ValueError, match="bounded limit of 64"):
        experiment.run_independent_ordered_impairment_snr_curve(
            snr_db=tuple(float(value) for value in range(65)),
            channel_seeds=(7,),
            blocks_per_channel=1,
        )
    with pytest.raises(TypeError, match="bounded sequence"):
        experiment.run_independent_ordered_impairment_snr_curve(snr_db="0,10")
    with pytest.raises(ValueError, match="nonnegative integers"):
        experiment.run_independent_ordered_impairment_snr_curve(snr_db=(0.0,), channel_seeds=([],))
    with pytest.raises(TypeError, match="channel_estimator"):
        experiment.run_independent_ordered_impairment_snr_curve(snr_db=(0.0,), channel_estimator=[])

    class OversizedSeedRoster(SequenceABC):
        def __len__(self) -> int:
            return 10**12

        def __getitem__(self, index):
            raise AssertionError("oversized seed roster must be rejected before indexing")

        def __iter__(self):
            raise AssertionError("oversized seed roster must be rejected before iteration")

    with pytest.raises(ValueError, match="exceeding the bounded limit"):
        experiment.run_independent_ordered_impairment_snr_curve(
            snr_db=(0.0, 1.0),
            channel_seeds=OversizedSeedRoster(),
            blocks_per_channel=8,
        )


def test_independent_snr_curve_bound_result_is_diagnostic_only(monkeypatch) -> None:
    monkeypatch.setattr(
        experiment,
        "run_independent_ordered_impairment_probe",
        _fake_snr_point,
    )
    report = experiment.run_independent_ordered_impairment_snr_curve(
        snr_db=(0.0,), channel_seeds=(3,), blocks_per_channel=1
    )

    result = build_hash_bound_result(report, created_utc="2026-10-26T00:00:00+00:00")

    assert result["interpretation"]["status"] == "diagnostic_only"
    assert result["interpretation"]["deployment_qualification"] is False
    assert experiment.verify_hash_bound_result(result)


def test_independent_snr_curve_auditor_recomputes_retained_crc_counts(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_probe(**kwargs):
        return _fake_snr_point(**kwargs)

    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", fake_probe)
    report = experiment.run_independent_ordered_impairment_snr_curve(
        snr_db=(0.0, 10.0), channel_seeds=(31, 37), blocks_per_channel=2
    )
    result = build_hash_bound_result(report, created_utc="2026-10-26T00:00:00+00:00")
    result_path = tmp_path / "curve.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    audit = audit_result_file(result_path)

    assert audit["status"] == "AUDIT_PARTIAL"
    assert audit["checks"]["paired_roster_noise_and_crc_summaries"] is True
    assert audit["checks"]["channel_cluster_curve_inference"] is True

    tampered_report = json.loads(json.dumps(report))
    tampered_report["curve_inference"]["by_paired_contrast"][0][
        "mean_channel_macro_bler_improvement"
    ] += 0.01
    tampered_result = build_hash_bound_result(
        tampered_report, created_utc="2026-10-26T00:00:00+00:00"
    )
    tampered_path = tmp_path / "tampered-curve.json"
    tampered_path.write_text(json.dumps(tampered_result), encoding="utf-8")

    tampered_audit = audit_result_file(tampered_path)

    assert tampered_audit["checks"]["result_digest"] is True
    assert tampered_audit["checks"]["channel_cluster_curve_inference"] is False


def test_independent_snr_curve_auditor_recomputes_all_condition_waveforms(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "curve.waveforms.npz"
    report = run_independent_ordered_impairment_snr_curve(
        snr_db=(-2.0, 4.0),
        channel_seeds=(74_211, 74_212),
        blocks_per_channel=1,
        waveform_evidence_output=evidence_path,
    )
    result_path = tmp_path / "curve.json"
    result_path.write_text(
        json.dumps(build_hash_bound_result(report, created_utc="2026-10-26T00:00:00+00:00")),
        encoding="utf-8",
    )

    audit = audit_result_file(result_path)

    assert audit["checks"]["waveform_equation_recomputation"] is True


def test_probe_decoder_selection_binds_finite_delay_estimator_source(monkeypatch) -> None:
    from experiments import g5_cp_bounded_dft_estimator as estimator_module

    marker = object()
    captured: dict[str, object] = {}

    def make_decoder(receiver, *, min_delay_samples, max_delay_samples):
        captured["receiver"] = receiver
        captured["min_delay_samples"] = min_delay_samples
        captured["max_delay_samples"] = max_delay_samples
        return marker

    monkeypatch.setattr(estimator_module, "make_finite_delay_dft_decoder", make_decoder)
    receiver = type("FakeReceiver", (), {"decode_pusch": object()})()

    decoder, identity = experiment._configure_probe_decoder(receiver, "finite_delay_dft")

    assert decoder is marker
    assert captured == {
        "receiver": receiver,
        "min_delay_samples": 0,
        "max_delay_samples": 7,
    }
    estimator_source = (
        Path(experiment.__file__).resolve().with_name("g5_cp_bounded_dft_estimator.py")
    )
    assert identity == {
        "name": "finite_delay_dft",
        "min_delay_samples": 0,
        "max_delay_samples": 7,
        "implementation_path": "experiments/g5_cp_bounded_dft_estimator.py",
        "implementation_sha256": hashlib.sha256(estimator_source.read_bytes()).hexdigest(),
    }


def test_probe_decoder_selection_keeps_default_ls_and_rejects_unknown_modes() -> None:
    receiver = type("FakeReceiver", (), {"decode_pusch": object()})()

    decoder, identity = experiment._configure_probe_decoder(receiver, "sionna_default_ls")

    assert decoder is receiver.decode_pusch
    assert identity == {"name": "sionna_default_ls"}
    with pytest.raises(ValueError, match="channel_estimator must be"):
        experiment._configure_probe_decoder(receiver, "unrecognized")


def test_independent_probe_schema_versions_bind_estimator_and_waveform_retention() -> None:
    assert experiment._independent_report_schema("sionna_default_ls", False).endswith(".v15")
    assert experiment._independent_report_schema("sionna_default_ls", True).endswith(".v19")
    assert experiment._independent_report_schema("finite_delay_dft", False).endswith(".v17")
    assert experiment._independent_report_schema("finite_delay_dft", True).endswith(".v20")
    assert experiment._independent_report_schema(
        "sionna_default_ls", False, device="cuda:0"
    ).endswith(".v21")
    assert experiment._independent_report_schema(
        "sionna_default_ls", True, device="cuda:0"
    ).endswith(".v22")
    assert experiment._independent_report_schema(
        "finite_delay_dft", False, device="cuda:0"
    ).endswith(".v23")
    assert experiment._independent_report_schema(
        "finite_delay_dft", True, device="cuda:0"
    ).endswith(".v24")


def test_independent_probe_rejects_unsupported_device_before_receiver_setup() -> None:
    with pytest.raises(ValueError, match="device must be cpu or"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(1,), blocks_per_channel=1, device="gpu"
        )


def test_independent_probe_rejects_malformed_cuda_device_before_receiver_setup() -> None:
    with pytest.raises(ValueError, match="device must be cpu or"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(1,), blocks_per_channel=1, device="cuda"
        )


def test_independent_probe_rejects_unavailable_cuda_before_receiver_setup(monkeypatch) -> None:
    monkeypatch.setattr(experiment.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no available CUDA device"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(1,), blocks_per_channel=1, device="cuda:0"
        )


def test_bounded_npz_writer_streams_channel_chunks_without_array_stacking() -> None:
    output = experiment._BoundedBytesIO(64 * 1024)
    tx_chunks = [
        np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
        np.asarray([5.0 + 6.0j, 7.0 + 8.0j], dtype=np.complex64),
    ]
    phase_chunks = [
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.asarray([0.3, 0.4], dtype=np.float32),
    ]

    experiment._write_bounded_npz(
        output,
        channel_count=2,
        channel_arrays={"tx_waveform": tx_chunks, "phase": phase_chunks},
        scalar_arrays={"channel_seeds": np.asarray([11, 12], dtype=np.int64)},
    )

    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["tx_waveform"], np.stack(tx_chunks))
        np.testing.assert_array_equal(archive["phase"], np.stack(phase_chunks))
        np.testing.assert_array_equal(archive["channel_seeds"], [11, 12])


def test_bounded_npz_writer_rejects_uncompressed_budget_overrun(monkeypatch) -> None:
    monkeypatch.setattr(experiment, "_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES", 64)

    with pytest.raises(ValueError, match="uncompressed waveform evidence"):
        experiment._write_bounded_npz(
            experiment._BoundedBytesIO(4096),
            channel_count=1,
            channel_arrays={"tx_waveform": [np.ones(128, dtype=np.complex64)]},
            scalar_arrays={"channel_seeds": np.asarray([1], dtype=np.int64)},
        )


def test_bounded_npz_writer_rejects_compressed_budget_overrun() -> None:
    with pytest.raises(ValueError, match="compressed waveform evidence"):
        experiment._write_bounded_npz(
            experiment._BoundedBytesIO(64),
            channel_count=1,
            channel_arrays={"tx_waveform": [np.ones(128, dtype=np.complex64)]},
            scalar_arrays={"channel_seeds": np.asarray([1], dtype=np.int64)},
        )


@pytest.mark.parametrize("name", ["", "../escape", "has space", "é", "n" * 129])
def test_bounded_npz_writer_rejects_unsafe_member_names(name: str) -> None:
    with pytest.raises(ValueError, match="short safe identifiers"):
        experiment._write_bounded_npz(
            experiment._BoundedBytesIO(4096),
            channel_count=1,
            channel_arrays={name: [np.asarray([1], dtype=np.int64)]},
            scalar_arrays={},
        )


def test_bounded_npz_writer_rejects_excessive_member_count() -> None:
    channel_arrays = {
        f"member_{index}": [np.asarray([index], dtype=np.int64)] for index in range(33)
    }
    with pytest.raises(ValueError, match="too many NPZ members"):
        experiment._write_bounded_npz(
            experiment._BoundedBytesIO(64 * 1024),
            channel_count=1,
            channel_arrays=channel_arrays,
            scalar_arrays={},
        )


def test_waveform_evidence_destination_is_checked_before_receiver_setup(
    monkeypatch, tmp_path: Path
) -> None:
    def unexpected_receiver(*_args, **_kwargs):
        raise AssertionError("destination validation must happen before receiver setup")

    monkeypatch.setattr(experiment, "SionnaPUSCHLSReceiver", unexpected_receiver)
    with pytest.raises(ValueError, match="sibling-safe .npz"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(1,),
            blocks_per_channel=1,
            waveform_evidence_output=tmp_path / "unsafe.json",
        )

    existing = tmp_path / "already-there.npz"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite waveform evidence"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(1,),
            blocks_per_channel=1,
            waveform_evidence_output=existing,
        )
    assert existing.read_bytes() == b"preserve"


@pytest.mark.native_sionna
def test_known_ordered_impairments_match_reference_and_recover_pusch_payloads():
    """A small independent-seed fixture exercises the frozen acceptance contract."""
    report = run_ordered_impairment_probe(payload_seeds=(74_101, 74_102), blocks_per_seed=8)

    assert report["acceptance"]["pass"] is True
    assert report["acceptance"]["observed"]["max_abs_timing_error_samples"] == 0
    assert report["acceptance"]["observed"]["max_samplewise_reference_absolute_error"] < 2e-5
    assert report["acceptance"]["observed"]["min_wrong_order_rmse"] > 1e-4
    assert report["paired_outcomes"]["clean"]["crc_failures"] == 0
    assert report["config"]["tx_iq_gain_imbalance_db"] == 0.5
    assert report["config"]["tx_iq_phase_imbalance_rad"] == 0.02
    assert report["config"]["rx_iq_gain_imbalance_db"] == 0.5
    assert report["config"]["rx_iq_phase_imbalance_rad"] == -0.02
    criteria = report["acceptance"]["criteria_frozen_before_run"]
    assert criteria["p95_abs_cfo_error_hz"] == 250.0
    assert criteria["max_abs_cfo_error_hz"] == 350.0
    assert criteria["max_impaired_corrected_bler"] == 0.10


def test_probe_restores_torch_thread_count_when_probe_raises(monkeypatch):
    original_threads = torch.get_num_threads()

    def fail_probe(**kwargs):
        del kwargs
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(experiment, "_run_ordered_impairment_probe", fail_probe)
    with pytest.raises(RuntimeError, match="synthetic probe failure"):
        run_ordered_impairment_probe(payload_seeds=(1,), blocks_per_seed=1)

    assert torch.get_num_threads() == original_threads


@pytest.mark.native_sionna
def test_ordered_impairment_probe_preserves_channel_level_pairing(tmp_path):
    report = run_independent_ordered_impairment_probe(
        channel_seeds=(74_201, 74_202),
        blocks_per_channel=2,
        waveform_evidence_output=tmp_path / "receiver.waveforms.npz",
    )

    assert report["schema"] == "umcd.g5_ordered_impairments_independent_channels.v19"
    receiver_input_digests = report["config"]["receiver_input_sha256_by_condition"]
    assert set(receiver_input_digests) == {"clean", "uncorrected", "corrected"}
    assert all(len(digests) == 2 for digests in receiver_input_digests.values())
    crc_tail_diagnostics = report["acceptance"]["inference"]["crc_channel_failure_diagnostics"]
    assert [row["condition"] for row in crc_tail_diagnostics] == [
        "clean",
        "corrected",
        "uncorrected",
    ]
    assert all(row["independent_channels"] == 2 for row in crc_tail_diagnostics)
    assert all(row["blocks_per_channel"] == 2 for row in crc_tail_diagnostics)
    assert all(
        0.0 <= row["exact_one_sided_95_percent_channel_prevalence_upper_bound"] <= 1.0
        for row in crc_tail_diagnostics
    )
    jsonschema = pytest.importorskip("jsonschema")
    evidence_schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-waveform-evidence.v3.schema.json"
    )
    evidence_schema = json.loads(evidence_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(evidence_schema).validate(report["config"]["waveform_evidence"])
    assert report["inference_scope"] == "the declared seeded synthetic channel family only"
    assert report["config"]["channel_seeds"] == [74_201, 74_202]
    assert report["config"]["timing_cp_guard_samples"] == 2
    assert report["config"]["sync_estimation_scope"] == "shared_per_channel_over_repeated_blocks"
    assert (
        report["config"]["cfo_cp_guard_samples"] == report["config"]["echo_delay_samples_range"][1]
    )
    assert len(report["config"]["per_channel_draws"]) == 2
    assert report["config"]["receiver_configuration"]["device"] == "cpu"
    assert report["config"]["receiver_configuration"]["n_size_bwp"] == 6
    assert (
        report["config"]["decoder_configuration"]["channel_estimator"]
        == report["config"]["channel_estimator"]
    )
    assert report["paired_outcomes"]["clean"]["blocks"] == 4
    assert len(report["timing_error_samples_by_block"]) == 4
    assert len(report["cfo_error_hz_by_block"]) == 4
    assert (
        len(
            report["acceptance"]["inference"][
                "paired_corrected_minus_clean_channel_bootstrap_95_percent_interval"
            ]
        )
        == 2
    )
    for channel in report["config"]["per_channel_draws"]:
        assert set(channel["crc_pass_by_condition"]) == {"clean", "uncorrected", "corrected"}
        assert len(channel["timing_error_samples"]) == 2
        assert len(channel["cfo_error_hz"]) == 2
        assert all(len(outcomes) == 2 for outcomes in channel["crc_pass_by_condition"].values())
        assert [block["block_index"] for block in channel["transport_blocks"]] == [0, 1]
        assert [block["payload_seed"] for block in channel["transport_blocks"]] == [
            channel["payload_seed"],
            channel["payload_seed"],
        ]
        assert all(
            block["crc_pass_by_condition"][condition]
            == channel["crc_pass_by_condition"][condition][index]
            for index, block in enumerate(channel["transport_blocks"])
            for condition in ("clean", "corrected", "uncorrected")
        )
    for condition in ("clean", "corrected", "uncorrected"):
        expected_failures = sum(
            not passed
            for channel in report["config"]["per_channel_draws"]
            for passed in channel["crc_pass_by_condition"][condition]
        )
        summary = report["paired_outcomes"][condition]
        assert summary["crc_failures"] == expected_failures
        assert summary["bler"] == expected_failures / summary["blocks"]
    tail_diagnostics = report["acceptance"]["inference"]["cfo_channel_tail_diagnostics"]
    assert [entry["absolute_error_threshold_hz"] for entry in tail_diagnostics] == [250.0, 350.0]
    for entry in tail_diagnostics:
        expected = sum(
            any(
                abs(error) >= entry["absolute_error_threshold_hz"]
                for error in channel["cfo_error_hz"]
            )
            for channel in report["config"]["per_channel_draws"]
        )
        assert entry["channels_with_any_block_exceedance"] == expected
        assert entry["independent_channels"] == 2
        assert (
            entry["channels_with_any_block_exceedance"] / 2
            <= (entry["exact_one_sided_95_percent_channel_prevalence_upper_bound"])
            <= 1
        )
    assert (
        "not a separate acceptance threshold"
        in report["acceptance"]["inference"]["cfo_channel_tail_interpretation"]
    )
    assert report["acceptance"]["observed"]["max_samplewise_reference_absolute_error"] < 2e-5
    evidence_bytes = (tmp_path / "receiver.waveforms.npz").read_bytes()
    maximum_error, minimum_wrong_order_rmse = _recompute_waveform_diagnostics(
        report["config"], report["config"]["per_channel_draws"], evidence_bytes
    )
    assert maximum_error < 2e-5
    assert minimum_wrong_order_rmse > 1e-4
    result_path = tmp_path / "receiver.json"
    write_hash_bound_result(
        result_path,
        build_hash_bound_result(report, created_utc="2026-10-25T00:00:00+00:00"),
    )
    audit = audit_result_file(result_path)
    assert audit["checks"]["waveform_equation_recomputation"] is True
    assert audit["status"] == "VERIFIED_RESULT_AND_WORKTREE_SOURCE_PENDING_COMMIT"


def test_shared_sync_waveform_pooling_concatenates_blocks_without_reordering():
    waveform = torch.arange(2 * 1 * 1 * 4).reshape(2, 1, 1, 4).to(torch.complex64)

    pooled = experiment._pool_repeated_waveforms_for_shared_sync(waveform)

    assert tuple(pooled.shape) == (1, 1, 1, 8)
    assert torch.equal(pooled.reshape(-1), torch.arange(8).to(torch.complex64))


def test_cli_accepts_an_explicit_independent_channel_seed_range(monkeypatch):
    received = {}

    def fake_probe(**kwargs):
        received.update(kwargs)
        return {"schema": "fake"}

    monkeypatch.setattr(experiment, "run_independent_ordered_impairment_probe", fake_probe)
    monkeypatch.setattr(
        "sys.argv",
        [
            "g5_ordered_impairments_receiver.py",
            "--independent-channel-count",
            "3",
            "--channel-seed-start",
            "991001",
            "--blocks-per-seed",
            "8",
            "--channel-estimator",
            "finite_delay_dft",
        ],
    )

    experiment.main()

    assert received["channel_seeds"] == (991001, 991002, 991003)
    assert received["blocks_per_channel"] == 8
    assert received["channel_estimator"] == "finite_delay_dft"


def test_independent_ordered_impairment_probe_rejects_duplicate_channels():
    with pytest.raises(ValueError, match="nonempty and unique"):
        run_independent_ordered_impairment_probe(channel_seeds=(1, 1), blocks_per_channel=1)


def test_independent_ordered_impairment_probe_bounds_total_work_before_receiver_setup(
    monkeypatch,
):
    monkeypatch.setattr(experiment, "_MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS", 3)
    monkeypatch.setattr(
        experiment,
        "SionnaPUSCHLSReceiver",
        lambda *_args, **_kwargs: pytest.fail("workload must be rejected before receiver setup"),
    )

    with pytest.raises(ValueError, match="4 transport blocks.*bounded limit of 3"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(10, 11), blocks_per_channel=2
        )


def test_waveform_evidence_seed_storage_range_is_checked_before_receiver_setup(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        experiment,
        "SionnaPUSCHLSReceiver",
        lambda *_args, **_kwargs: pytest.fail("invalid seed must fail before receiver setup"),
    )

    with pytest.raises(ValueError, match="fit signed 64-bit storage"):
        experiment._run_independent_ordered_impairment_probe(
            channel_seeds=(np.iinfo(np.int64).max + 1,),
            blocks_per_channel=1,
            waveform_evidence_output=tmp_path / "waveforms.npz",
        )

"""Software-level contracts for standardized-channel PUSCH adapter boundaries."""

from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import integrations.sionna.standardized_pusch as standardized_pusch
import numpy as np
import pytest
from datasets.standardized import StandardizedChannelRun
from integrations.sionna.ofdm_impairments import NRWaveformImpairments
from receivers.nr import NRPUSCHConfig


def _validation_run(*, cfr_shape=(1, 1, 1, 1, 1, 8)):
    record = SimpleNamespace(
        cfr=np.zeros(cfr_shape, dtype=np.complex64),
        sample_id="fixture-sample",
    )
    run = object.__new__(StandardizedChannelRun)
    object.__setattr__(run, "records", (record,))
    return run


def _build(run, **overrides):
    arguments = {
        "snr_db": 10.0,
        "seed": 1,
        "receiver_config": NRPUSCHConfig(),
    }
    arguments.update(overrides)
    return standardized_pusch.build_standardized_pusch_dmrs_records(run, **arguments)


@pytest.mark.parametrize(
    ("run", "overrides", "error", "message"),
    [
        (object(), {}, TypeError, "StandardizedChannelRun"),
        (_validation_run(), {"snr_db": float("nan")}, ValueError, "snr_db must be finite"),
        (_validation_run(), {"seed": True}, ValueError, "seed must be an integer"),
        (_validation_run(), {"seed": -1}, ValueError, "seed must be an integer"),
        (_validation_run(), {"noise_seed": True}, ValueError, "noise_seed must be an integer"),
        (_validation_run(), {"noise_seed": 2**63 - 1}, ValueError, "noise_seed must be an integer"),
        (_validation_run(), {"impairments": object()}, TypeError, "NRWaveformImpairments"),
        (
            _validation_run(),
            {"impairments": object(), "impairment_seed": 1},
            TypeError,
            "NRWaveformImpairments",
        ),
        (_validation_run(), {"impairment_seed": 1}, ValueError, "provided together"),
        (_validation_run(), {"channel_mode": "invalid"}, ValueError, "channel_mode must"),
        (
            _validation_run(),
            {"impairment_seed": True},
            ValueError,
            "impairments and impairment_seed",
        ),
        (
            _validation_run(),
            {"impairments": NRWaveformImpairments(), "impairment_seed": True},
            ValueError,
            "impairment_seed must be an integer",
        ),
        (_validation_run(), {"time_index": True}, ValueError, "time_index must"),
        (_validation_run(), {"time_index": -1}, ValueError, "time_index must"),
        (_validation_run(), {"device": "tpu"}, ValueError, "device must be cpu or"),
        (_validation_run(), {"time_index": 8}, ValueError, "outside sample"),
        (
            _validation_run(cfr_shape=(1, 1, 2, 1, 1, 8)),
            {},
            ValueError,
            "one TX and one TX antenna",
        ),
        (_validation_run(cfr_shape=(2, 1, 1, 1, 1, 8)), {}, ValueError, "one RX node"),
    ],
)
def test_standardized_pusch_rejects_invalid_run_and_generation_contracts(
    run, overrides, error, message
):
    with pytest.raises(error, match=message):
        _build(run, **overrides)


def test_phase_noise_without_linewidth_is_identity_and_rejects_missing_node_axes():
    torch = pytest.importorskip("torch")
    waveform = torch.ones((2, 64), dtype=torch.complex64)
    assert (
        standardized_pusch._apply_phase_noise(
            waveform,
            linewidth_hz=None,
            seed=0,
            sampling_frequency_hz=1e6,
            device="cpu",
        )
        is waveform
    )
    with pytest.raises(ValueError, match="node and antenna axes"):
        standardized_pusch._apply_phase_noise(
            waveform,
            linewidth_hz=10.0,
            seed=0,
            sampling_frequency_hz=1e6,
            device="cpu",
        )


def _diagnostic_record(*, reference_bits=np.asarray([[0, 1, 1]], dtype=np.uint8)):
    reference = None if reference_bits is None else np.asarray(reference_bits, dtype=np.uint8)
    provenance = {
        "noise_variance_per_complex_dimension": 0.01,
        "reference_transport_block_shape": None if reference is None else tuple(reference.shape),
        "reference_transport_block_sha256": (
            None if reference is None else hashlib.sha256(reference.tobytes(order="C")).hexdigest()
        ),
    }
    return SimpleNamespace(
        observation=SimpleNamespace(
            sample_id="sample-1",
            received_grid=np.ones((1, 1, 1, 14, 48), dtype=np.complex64),
            channel_estimate=SimpleNamespace(
                h_hat=np.ones((1, 1, 1, 1, 1, 14, 48), dtype=np.complex64)
            ),
        ),
        provenance=provenance,
        targets={"channel_cfr": np.ones((1, 1, 1, 1, 1, 14, 48), dtype=np.complex64)},
        reference_transport_block_bits=reference,
    )


def _install_fake_diagnostic_decoder(monkeypatch, torch, *, decoded_bits=None, crc=None):
    phy = ModuleType("sionna.phy")
    phy.__path__ = []
    nr = ModuleType("sionna.phy.nr")
    if decoded_bits is None:
        decoded_bits = torch.tensor([[[0, 1, 0]]], dtype=torch.uint8)
    if crc is None:
        crc = torch.tensor([[True]], dtype=torch.bool)

    class Decoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            return decoded_bits, crc

    nr.PUSCHReceiver = Decoder
    monkeypatch.setitem(sys.modules, "sionna.phy", phy)
    monkeypatch.setitem(sys.modules, "sionna.phy.nr", nr)
    monkeypatch.setattr(
        standardized_pusch,
        "SionnaPUSCHLSReceiver",
        lambda *_args, **_kwargs: SimpleNamespace(_transmitter=object()),
    )


def test_standardized_pusch_diagnostic_audit_counts_payload_errors_and_crc(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_diagnostic_decoder(monkeypatch, torch)

    outcome = standardized_pusch.audit_standardized_pusch_records_for_diagnostic(
        (_diagnostic_record(),),
        receiver_config=NRPUSCHConfig(),
        estimate_source="dmrs_ls",
    )[0]

    assert outcome.sample_id == "sample-1"
    assert outcome.transport_block_crc_pass is True
    assert outcome.payload_bit_errors == 1
    assert outcome.payload_bit_count == 3


@pytest.mark.parametrize(
    ("records", "estimate_source", "message"),
    [
        ((), "dmrs_ls", "records must not be empty"),
        ((_diagnostic_record(),), "invalid", "unsupported diagnostic estimate source"),
    ],
)
def test_standardized_pusch_diagnostic_audits_validate_arguments(records, estimate_source, message):
    with pytest.raises(ValueError, match=message):
        standardized_pusch.audit_standardized_pusch_records_for_diagnostic(
            records,
            receiver_config=NRPUSCHConfig(),
            estimate_source=estimate_source,
        )


@pytest.mark.parametrize(
    "decode",
    [
        standardized_pusch.audit_standardized_pusch_records_for_diagnostic,
        standardized_pusch.decode_standardized_pusch_records_for_diagnostic,
    ],
)
def test_standardized_pusch_diagnostic_decode_validates_empty_and_estimator(decode):
    for records, source, message in (
        ((), "dmrs_ls", "records must not be empty"),
        ((_diagnostic_record(),), "invalid", "unsupported diagnostic estimate source"),
    ):
        with pytest.raises(ValueError, match=message):
            decode(records, receiver_config=NRPUSCHConfig(), estimate_source=source)


@pytest.mark.parametrize(
    "decode",
    [
        standardized_pusch.audit_standardized_pusch_records_for_diagnostic,
        standardized_pusch.decode_standardized_pusch_records_for_diagnostic,
    ],
)
def test_standardized_pusch_diagnostic_decode_rejects_unavailable_cuda(monkeypatch, decode):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA decoding was requested"):
        decode(
            (_diagnostic_record(),),
            receiver_config=NRPUSCHConfig(),
            estimate_source="dmrs_ls",
            device="cuda:0",
        )


@pytest.mark.parametrize(
    ("record", "decoded_bits", "message"),
    [
        (_diagnostic_record(reference_bits=None), None, "no retained TX reference"),
        (_diagnostic_record(), None, "decoded payload shape differs"),
        (_diagnostic_record(), "digest", "differs from its provenance digest"),
    ],
)
def test_standardized_pusch_diagnostic_audit_rejects_missing_or_drifted_reference(
    monkeypatch, record, decoded_bits, message
):
    torch = pytest.importorskip("torch")
    if decoded_bits == "digest":
        record.provenance["reference_transport_block_sha256"] = "0" * 64
    elif message == "decoded payload shape differs":
        _install_fake_diagnostic_decoder(
            monkeypatch, torch, decoded_bits=torch.zeros((1, 2, 3), dtype=torch.uint8)
        )
    else:
        _install_fake_diagnostic_decoder(monkeypatch, torch)
    if decoded_bits == "digest":
        _install_fake_diagnostic_decoder(monkeypatch, torch)
    with pytest.raises(ValueError, match=message):
        standardized_pusch.audit_standardized_pusch_records_for_diagnostic(
            (record,), receiver_config=NRPUSCHConfig(), estimate_source="diagonal_cfr_oracle"
        )


def test_legacy_standardized_pusch_decode_keeps_crc_only_records_supported(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_diagnostic_decoder(
        monkeypatch,
        torch,
        decoded_bits=torch.zeros((1, 1, 3), dtype=torch.uint8),
        crc=torch.tensor([[True]], dtype=torch.bool),
    )
    records = (_diagnostic_record(reference_bits=None),)

    assert standardized_pusch.decode_standardized_pusch_records_for_diagnostic(
        records, receiver_config=NRPUSCHConfig(), estimate_source="dmrs_ls"
    ) == (True,)

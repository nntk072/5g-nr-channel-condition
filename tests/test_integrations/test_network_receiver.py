import hashlib
import inspect
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.network_phy import (  # noqa: E402
    add_independent_receiver_waveform_awgn,
    apply_network_transmitter_iq_imbalance,
    compose_received_waveforms,
    propagate_network_transmit_waveforms,
)
from integrations.sionna.network_receiver import (  # noqa: E402
    NetworkPUSCHWorkflowResult,
    _real_noise_profile,
    _receiver_implementation_identity,
    _tensor_sha256,
    _unitary_sionna_noise_conversion,
    decode_noisy_network_pusch,
    run_network_pusch_from_transmitters,
    run_network_pusch_receiver,
)
from integrations.sionna.ofdm_impairments import apply_iq_imbalance  # noqa: E402


class _FakePUSCHReceiver:
    device = "cpu"

    metadata = {
        "receiver": "test_double",
        "resource_grid": {
            "num_ofdm_symbols": 2,
            "fft_size": 4,
            "cyclic_prefix_length_samples": 0,
            "subcarrier_spacing_hz": 1,
        },
    }

    def __init__(self):
        self.decoded = False
        self.noise_variance = None
        self.demodulator_waveform = None

    def demodulate_slot_waveform(self, waveform):
        assert waveform.shape == (1, 1, 1, 8)
        self.demodulator_waveform = waveform.clone()
        return torch.ones((1, 1, 1, 2, 4), dtype=torch.complex64, device=waveform.device)

    def decode_pusch(self, received_grid, noise_variance):
        self.decoded = True
        self.noise_variance = noise_variance.clone()
        return torch.zeros((1, 1, 12), dtype=torch.bool), torch.tensor([[True]])


def test_network_receiver_implementation_identity_rejects_uninspectable_methods():
    class BuiltinMethodReceiver(_FakePUSCHReceiver):
        demodulate_slot_waveform = math.sqrt

    with pytest.raises(ValueError, match="implementation source is unavailable"):
        _receiver_implementation_identity(BuiltinMethodReceiver())


def test_real_noise_profile_normalizes_integer_and_float_profiles():
    profile = _real_noise_profile(torch, (1, 0.25), rx_count=2, device="cpu")

    assert profile.dtype == torch.float64
    torch.testing.assert_close(profile, torch.tensor((1.0, 0.25), dtype=torch.float64))


@pytest.mark.parametrize(
    ("values", "rx_count", "message"),
    [
        (True, 1, "non-Boolean"),
        (object(), 1, "real numeric"),
        ([True], 1, "non-Boolean"),
        ([1 + 2j], 1, r"real \[rx\] vector"),
        (torch.ones(2, dtype=torch.float32).to_sparse(), 2, r"real \[rx\] vector"),
        ([1.0], 2, r"real \[rx\] vector"),
        ([float("nan")], 1, "finite and positive"),
        ([0.0], 1, "finite and positive"),
        ([-0.1], 1, "finite and positive"),
    ],
)
def test_real_noise_profile_rejects_invalid_values(values, rx_count, message):
    with pytest.raises(ValueError, match=message):
        _real_noise_profile(torch, values, rx_count=rx_count, device="cpu")


def _network_waveform():
    link_waveforms = torch.ones((1, 1, 1, 1, 8), dtype=torch.complex64)
    composition = compose_received_waveforms(
        link_waveforms,
        link_start_samples=((0,),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.02,), seed=19)
    return composition, noisy


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_network_transmitter_iq_imbalance_precedes_per_link_propagation(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    transmit = torch.tensor(
        [[[[1.0 + 2.0j, -0.5 + 0.25j, 0.2 - 0.7j]], [[0.3 - 0.2j, 1.5 + 0.5j, -0.1 + 1.1j]]]],
        dtype=torch.complex64,
        device=device,
    )
    gain_db = (0.7, -0.4)
    phase_rad = (0.08, -0.12)

    tx_stage = apply_network_transmitter_iq_imbalance(
        transmit,
        tx_iq_gain_imbalance_db=gain_db,
        tx_iq_phase_imbalance_rad=phase_rad,
    )

    expected_per_tx = torch.stack(
        [
            apply_iq_imbalance(
                transmit[:, tx_index],
                gain_imbalance_db=gain_db[tx_index],
                phase_imbalance_rad=phase_rad[tx_index],
            )
            for tx_index in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(tx_stage.waveform, expected_per_tx)
    assert tx_stage.signal_stage == "transmitter_iq_imbalance_before_propagation"
    assert tuple(tx_stage.tx_iq_gain_imbalance_db.shape) == (2,)
    assert tx_stage.tx_iq_gain_imbalance_db.tolist() == list(gain_db)
    assert tx_stage.tx_iq_phase_imbalance_rad.tolist() == list(phase_rad)

    channel = torch.tensor(
        [[[[[0.7 + 0.4j]], [[-0.2 + 0.9j]]]]], dtype=torch.complex64, device=device
    )
    propagated = tx_stage.waveform.unsqueeze(1) * channel
    composition = compose_received_waveforms(
        propagated,
        link_start_samples=((0, 0),),
        capture_num_samples=3,
        sample_rate_hz=30_720_000.0,
    )
    torch.testing.assert_close(composition.received_waveform, propagated.sum(dim=2))

    h = channel[0, 0, 0, 0, 0]
    tx_first_then_channel = tx_stage.waveform[:, 0, 0] * h
    channel_first_then_tx = apply_iq_imbalance(
        transmit[:, 0, 0] * h,
        gain_imbalance_db=gain_db[0],
        phase_imbalance_rad=phase_rad[0],
    )
    assert not torch.allclose(tx_first_then_channel, channel_first_then_tx)


def test_transmitter_to_pusch_workflow_applies_iq_before_declared_mimo_fir():
    transmit = torch.ones((1, 1, 1, 8), dtype=torch.complex64)
    taps = torch.tensor([[[[[1.0 + 0.5j]]]]], dtype=torch.complex64)
    result = run_network_pusch_from_transmitters(
        transmit,
        taps,
        _FakePUSCHReceiver(),
        tx_iq_gain_imbalance_db=(6.0,),
        tx_iq_phase_imbalance_rad=(0.0,),
        link_start_samples=((0,),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
        noise_variance_w_per_sample=(0.0,),
        noise_seed=19,
        noise_variance_w_per_re=(0.04,),
    )

    expected_tx = 10 ** (6.0 / 40.0)
    torch.testing.assert_close(
        result.propagation.link_waveforms,
        torch.full((1, 1, 1, 1, 8), expected_tx * (1.0 + 0.5j), dtype=torch.complex64),
    )
    assert result.transmitter_stage.signal_stage == "transmitter_iq_imbalance_before_propagation"
    assert result.report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert result.report["dimensions"]["fir_taps"] == 1
    assert (
        result.report["receiver_workflow_report_sha256"]
        == result.receiver_workflow.report["report_sha256"]
    )
    assert len(result.report["report_sha256"]) == 64


def test_transmitter_to_pusch_workflow_preflights_total_memory_before_tx_iq(monkeypatch):
    def fail_if_applied(*args, **kwargs):
        pytest.fail("TX IQ must not run before the end-to-end allocation preflight")

    monkeypatch.setattr(
        "integrations.sionna.network_receiver.apply_network_transmitter_iq_imbalance",
        fail_if_applied,
    )
    with pytest.raises(ValueError, match="before TX IQ"):
        run_network_pusch_from_transmitters(
            torch.ones((1, 1, 1, 8), dtype=torch.complex64),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            _FakePUSCHReceiver(),
            tx_iq_gain_imbalance_db=(0.0,),
            tx_iq_phase_imbalance_rad=(0.0,),
            link_start_samples=((0,),),
            capture_num_samples=8,
            sample_rate_hz=4.0,
            noise_variance_w_per_sample=(0.02,),
            noise_seed=19,
            noise_variance_w_per_re=(0.04,),
            max_tensor_elements=1,
        )


def test_propagate_network_transmitter_waveforms_preserves_full_fir_tail():
    tx_stage = apply_network_transmitter_iq_imbalance(
        torch.tensor([[[[1.0 + 0j, 2.0 + 0j]]]], dtype=torch.complex64),
        tx_iq_gain_imbalance_db=(0.0,),
        tx_iq_phase_imbalance_rad=(0.0,),
    )
    taps = torch.tensor([[[[[1.0 + 0j, 0.5 + 0j]]]]], dtype=torch.complex64)

    result = propagate_network_transmit_waveforms(tx_stage, taps)

    torch.testing.assert_close(
        result.link_waveforms,
        torch.tensor([[[[[1.0 + 0j, 2.5 + 0j, 1.0 + 0j]]]]], dtype=torch.complex64),
    )


@pytest.mark.parametrize(
    ("gain", "phase", "error"),
    [
        ((0.0,), (0.0, 0.0), "tx_iq_gain_imbalance_db must have shape"),
        ((False, 0.0), (0.0, 0.0), "tx_iq_gain_imbalance_db must contain real numeric values"),
        ((0.0, 0.0), (0.0, float("inf")), "tx_iq_phase_imbalance_rad must be finite"),
    ],
)
def test_network_transmitter_iq_imbalance_rejects_invalid_node_profiles(gain, phase, error):
    waveform = torch.ones((1, 2, 1, 4), dtype=torch.complex64)

    with pytest.raises(ValueError, match=error):
        apply_network_transmitter_iq_imbalance(
            waveform,
            tx_iq_gain_imbalance_db=gain,
            tx_iq_phase_imbalance_rad=phase,
        )


def test_network_waveform_reaches_pusch_decoder_with_bound_inputs_and_status():
    composition, noisy = _network_waveform()
    receiver = _FakePUSCHReceiver()

    result = decode_noisy_network_pusch(
        composition,
        noisy,
        receiver,
        noise_variance_w_per_re=(0.04,),
    )

    assert receiver.decoded
    torch.testing.assert_close(receiver.noise_variance, torch.tensor([[[0.04]]]))
    assert result.received_grid.shape == (1, 1, 1, 2, 4)
    assert bool(result.crc_ok.all())
    assert result.report["schema"] == "umcd.network_pusch_receiver_decode.v6"
    assert result.report["status"] == "OBSERVED_NOT_QUALIFIED"
    receiver_implementation = result.report["receiver_implementation"]
    assert receiver_implementation["qualified_class_name"].endswith("._FakePUSCHReceiver")
    assert receiver_implementation["method_sha256"] == {
        name: hashlib.sha256(
            inspect.getsource(getattr(_FakePUSCHReceiver, name)).encode()
        ).hexdigest()
        for name in ("demodulate_slot_waveform", "decode_pusch")
    }
    assert result.report["receiver_noise_convention"] == "watts_per_resource_element"
    assert result.report["noise_variance_conversion"] == "caller_supplied_per_re_variance_v1"
    assert result.report["input_sha256"]["demodulator_waveform"] == _tensor_sha256(
        torch, receiver.demodulator_waveform
    )
    assert result.report["shape"]["transport_blocks"] == 1
    cp_diagnostic = result.report["declared_fir_cp_diagnostic"]
    assert cp_diagnostic["maximum_fir_memory_samples"] == 0
    assert cp_diagnostic["exceeds_cyclic_prefix"] is False
    assert cp_diagnostic["maximum_composed_support_span_samples"] == 0
    assert cp_diagnostic["composed_support_exceeds_cyclic_prefix"] is False
    assert cp_diagnostic["interpretation"].endswith("OR_QUALIFICATION")
    assert len(result.report["report_sha256"]) == 64
    report_body = dict(result.report)
    report_sha256 = report_body.pop("report_sha256")
    assert (
        report_sha256
        == hashlib.sha256(
            json.dumps(report_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def test_network_pusch_workflow_composes_adds_noise_decodes_and_binds_replay():
    link_waveforms = torch.ones((1, 1, 1, 1, 8), dtype=torch.complex64)
    kwargs = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 8,
        "sample_rate_hz": 4.0,
        "noise_variance_w_per_sample": (0.02,),
        "noise_seed": 19,
        "noise_variance_w_per_re": (0.04,),
    }
    first = run_network_pusch_receiver(link_waveforms, _FakePUSCHReceiver(), **kwargs)
    second = run_network_pusch_receiver(link_waveforms, _FakePUSCHReceiver(), **kwargs)

    assert isinstance(first, NetworkPUSCHWorkflowResult)
    assert first.report["schema"] == "umcd.network_pusch_receiver_workflow.v1"
    assert first.report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert first.report["noise_seed"] == 19
    assert first.report["report_sha256"] == second.report["report_sha256"]
    assert first.report["decode_report_sha256"] == first.decode.report["report_sha256"]
    report_body = dict(first.report)
    report_sha256 = report_body.pop("report_sha256")
    assert (
        report_sha256
        == hashlib.sha256(
            json.dumps(report_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    assert set(first.report["implementation_sha256"]) == {
        "workflow",
        "waveform_composition",
        "receiver_awgn",
        "pusch_decode",
    }
    assert first.report["boundary"] == (
        "SOFTWARE_REPLAY_ONLY_NOT_RECEIVER_OR_CHANNEL_QUALIFICATION"
    )
    assert first.report["input_sha256"]["noise_waveform"] == _tensor_sha256(
        torch, first.noisy_waveform.noise_waveform
    )


def test_network_pusch_workflow_preflights_capture_memory_before_composition(monkeypatch):
    def fail_if_composed(*args, **kwargs):
        pytest.fail("waveform composition must not run before the memory preflight")

    monkeypatch.setattr(
        "integrations.sionna.network_receiver.compose_received_waveforms", fail_if_composed
    )
    with pytest.raises(ValueError, match="exceeds max_tensor_elements before composition"):
        run_network_pusch_receiver(
            torch.ones((1, 1, 1, 1, 8), dtype=torch.complex64),
            _FakePUSCHReceiver(),
            link_start_samples=((0,),),
            capture_num_samples=8,
            sample_rate_hz=4.0,
            noise_variance_w_per_sample=(0.02,),
            noise_seed=19,
            noise_variance_w_per_re=(0.04,),
            max_tensor_elements=16,
        )


def test_network_receiver_hashes_the_complex64_waveform_actually_demodulated():
    links = torch.full((1, 1, 1, 1, 8), 1.0 + 2.0**-30 + 0.25j, dtype=torch.complex128)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.0,), seed=19)
    receiver = _FakePUSCHReceiver()

    result = decode_noisy_network_pusch(
        composition, noisy, receiver, noise_variance_w_per_re=(0.04,)
    )

    inputs = result.report["input_sha256"]
    assert noisy.received_waveform.dtype == torch.complex128
    assert receiver.demodulator_waveform.dtype == torch.complex64
    assert inputs["noisy_received_waveform"] == _tensor_sha256(torch, noisy.received_waveform)
    assert inputs["demodulator_waveform"] == _tensor_sha256(torch, receiver.demodulator_waveform)
    assert inputs["demodulator_waveform"] != inputs["noisy_received_waveform"]


def test_network_receiver_resource_preflight_counts_converted_demodulator_waveform():
    links = torch.ones((1, 1, 1, 1, 8), dtype=torch.complex128)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.0,), seed=19)
    receiver = _FakePUSCHReceiver()

    with pytest.raises(ValueError, match="receiver waveform.*max_tensor_elements"):
        decode_noisy_network_pusch(
            composition,
            noisy,
            receiver,
            noise_variance_w_per_re=(0.04,),
            max_tensor_elements=40,
        )

    assert receiver.demodulator_waveform is None
    assert not receiver.decoded


@pytest.mark.parametrize(
    "decoded_bits",
    [
        pytest.param(torch.full((1, 1, 12), 0.5), id="fractional-float"),
        pytest.param(torch.full((1, 1, 12), 2, dtype=torch.int32), id="integer-out-of-range"),
        pytest.param(torch.zeros((1, 1, 12), dtype=torch.complex64), id="complex"),
        pytest.param(torch.tensor(0.0), id="scalar"),
        pytest.param(
            torch.sparse_coo_tensor([[0]], [0.0], size=(1,), check_invariants=True),
            id="sparse",
        ),
    ],
)
def test_network_receiver_rejects_nonbinary_decoder_outputs(decoded_bits):
    class InvalidBitReceiver(_FakePUSCHReceiver):
        def decode_pusch(self, received_grid, noise_variance):
            self.decoded = True
            return decoded_bits, torch.tensor([[True]])

    composition, noisy = _network_waveform()
    receiver = InvalidBitReceiver()

    with pytest.raises(ValueError, match="dense binary batch-aligned bits"):
        decode_noisy_network_pusch(composition, noisy, receiver, noise_variance_w_per_re=(0.04,))


def test_network_receiver_requires_crc_shape_to_match_the_decoded_transport_block_axes():
    class MismatchedCRCReceiver(_FakePUSCHReceiver):
        def decode_pusch(self, received_grid, noise_variance):
            self.decoded = True
            return torch.zeros((1, 2, 12), dtype=torch.bool), torch.tensor([[True]])

    composition, noisy = _network_waveform()
    receiver = MismatchedCRCReceiver()

    with pytest.raises(ValueError, match="dense binary batch-aligned bits"):
        decode_noisy_network_pusch(composition, noisy, receiver, noise_variance_w_per_re=(0.04,))


def test_network_receiver_reports_declared_fir_memory_relative_to_cyclic_prefix():
    links = torch.ones((1, 1, 1, 1, 8), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((3,),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
        link_filter_taps=torch.tensor([[[1.0 + 0j, 0.0 + 0j, 0.5 + 0j]]]),
        link_fractional_delay_samples=((0.5,),),
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.02,), seed=19)
    receiver = _FakePUSCHReceiver()

    result = decode_noisy_network_pusch(
        composition,
        noisy,
        receiver,
        noise_variance_w_per_re=(0.04,),
    )

    diagnostic = result.report["declared_fir_cp_diagnostic"]
    assert diagnostic["per_receiver_link_support"] == [
        [{"first_nonzero_tap": 0, "last_nonzero_tap": 2, "memory_samples": 2}]
    ]
    assert diagnostic["maximum_fir_memory_samples"] == 2
    assert diagnostic["cyclic_prefix_samples"] == 0
    assert diagnostic["exceeds_cyclic_prefix"] is True
    assert diagnostic["per_receiver_composed_support"] == [
        {
            "first_effective_support_sample_relative_to_capture": -12.5,
            "last_effective_support_sample_relative_to_capture": 21.5,
            "span_upper_bound_samples": 35,
        }
    ]
    assert diagnostic["maximum_composed_support_span_samples"] == 35
    assert diagnostic["composed_support_exceeds_cyclic_prefix"] is True
    assert len(diagnostic["filter_taps_sha256"]) == 64


def test_network_receiver_cp_diagnostic_includes_relative_link_arrival_offsets():
    links = torch.ones((1, 1, 2, 1, 8), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0, 4),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.02,), seed=19)
    result = decode_noisy_network_pusch(
        composition,
        noisy,
        _FakePUSCHReceiver(),
        noise_variance_w_per_re=(0.04,),
    )
    diagnostic = result.report["declared_fir_cp_diagnostic"]
    assert diagnostic["per_receiver_composed_support"] == [
        {
            "first_effective_support_sample_relative_to_capture": 0.0,
            "last_effective_support_sample_relative_to_capture": 4.0,
            "span_upper_bound_samples": 4,
        }
    ]
    assert diagnostic["maximum_composed_support_span_samples"] == 4
    assert diagnostic["composed_support_exceeds_cyclic_prefix"] is True


def test_network_receiver_counts_retained_fir_taps_in_tensor_resource_limit():
    links = torch.ones((1, 1, 1, 1, 8), dtype=torch.complex64)
    taps = torch.ones((1, 1, 128), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=8,
        sample_rate_hz=4.0,
        link_filter_taps=taps,
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.02,), seed=19)
    receiver = _FakePUSCHReceiver()

    with pytest.raises(ValueError, match="network waveform inputs exceed max_tensor_elements"):
        decode_noisy_network_pusch(
            composition,
            noisy,
            receiver,
            noise_variance_w_per_re=(0.04,),
            max_tensor_elements=40,
        )
    assert not receiver.decoded


def test_network_receiver_rejects_broken_source_sum_and_noise_lineage_before_decode():
    composition, noisy = _network_waveform()
    receiver = _FakePUSCHReceiver()
    invalid_composition = replace(
        composition, received_waveform=composition.received_waveform + 1.0
    )

    with pytest.raises(ValueError, match="do not sum"):
        decode_noisy_network_pusch(
            invalid_composition,
            noisy,
            receiver,
            noise_variance_w_per_re=(0.04,),
        )
    assert not receiver.decoded

    invalid_noisy = replace(noisy, received_waveform=noisy.received_waveform + 1.0)
    with pytest.raises(ValueError, match="does not equal source waveform plus noise"):
        decode_noisy_network_pusch(
            composition,
            invalid_noisy,
            receiver,
            noise_variance_w_per_re=(0.04,),
        )
    assert not receiver.decoded


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("composition", "received_waveform", 1.0, "network waveforms must have"),
        (
            "composition",
            "received_waveform",
            torch.ones((1, 1, 1, 8)),
            "network waveforms must have",
        ),
        (
            "composition",
            "received_waveform",
            torch.ones((1, 1, 8), dtype=torch.complex64),
            "network waveforms must have",
        ),
        (
            "composition",
            "source_contributions",
            torch.ones((1, 1, 1, 8), dtype=torch.complex64),
            "network waveforms must have",
        ),
        (
            "composition",
            "source_contributions",
            torch.ones((1, 2, 1, 2, 8), dtype=torch.complex64),
            "network waveforms must have",
        ),
        (
            "noisy",
            "noise_waveform",
            torch.ones((1, 1, 1, 8)),
            "network waveforms must have",
        ),
        (
            "noisy",
            "received_waveform",
            torch.ones((1, 1, 1, 8)),
            "network waveforms must have",
        ),
        (
            "noisy",
            "noise_waveform",
            torch.zeros((1, 1, 1, 7), dtype=torch.complex64),
            "axes must be nonempty",
        ),
        (
            "noisy",
            "received_waveform",
            torch.zeros((1, 1, 1, 7), dtype=torch.complex64),
            "must match the source composition shape",
        ),
        (
            "composition",
            "received_waveform",
            torch.full((1, 1, 1, 8), float("inf"), dtype=torch.complex64),
            "must be finite",
        ),
    ],
)
def test_network_receiver_rejects_malformed_waveform_records(target, field, value, message):
    composition, noisy = _network_waveform()
    record = composition if target == "composition" else noisy
    invalid_record = replace(record, **{field: value})
    if target == "composition":
        composition = invalid_record
    else:
        noisy = invalid_record

    with pytest.raises(ValueError, match=message):
        decode_noisy_network_pusch(
            composition,
            noisy,
            _FakePUSCHReceiver(),
            noise_variance_w_per_re=(0.04,),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"resource_grid": []},
        {"resource_grid": {"fft_size": 4}, "invalid": float("inf")},
        {"resource_grid": {"fft_size": True}},
        {"resource_grid": {"fft_size": 0}},
    ],
)
def test_network_receiver_rejects_invalid_receiver_metadata(metadata):
    class InvalidMetadataReceiver(_FakePUSCHReceiver):
        pass

    InvalidMetadataReceiver.metadata = metadata
    composition, noisy = _network_waveform()

    with pytest.raises(ValueError, match="receiver"):
        decode_noisy_network_pusch(
            composition,
            noisy,
            InvalidMetadataReceiver(),
            noise_variance_w_per_re=(0.04,),
        )


@pytest.mark.parametrize("device", [None, ""])
def test_network_receiver_requires_declared_device(device):
    class MissingDeviceReceiver(_FakePUSCHReceiver):
        pass

    MissingDeviceReceiver.device = device
    composition, noisy = _network_waveform()

    with pytest.raises(ValueError, match="execution device"):
        decode_noisy_network_pusch(
            composition,
            noisy,
            MissingDeviceReceiver(),
            noise_variance_w_per_re=(0.04,),
        )


@pytest.mark.parametrize(
    "filter_taps",
    [
        torch.ones((1, 1, 1, 1), dtype=torch.complex64),
        torch.ones((1, 2, 1), dtype=torch.complex64),
        torch.empty((1, 1, 0), dtype=torch.complex64),
    ],
)
def test_network_receiver_rejects_invalid_link_filter_taps(filter_taps):
    composition, noisy = _network_waveform()

    with pytest.raises(ValueError, match="link FIR taps"):
        decode_noisy_network_pusch(
            replace(composition, link_filter_taps=filter_taps),
            noisy,
            _FakePUSCHReceiver(),
            noise_variance_w_per_re=(0.04,),
        )


@pytest.mark.parametrize(
    "variance",
    [
        (0.02,),
        torch.tensor((0.02, 0.03)),
        torch.tensor((-0.02,)),
    ],
)
def test_network_receiver_requires_valid_sample_noise_variance_vector(variance):
    composition, noisy = _network_waveform()
    with pytest.raises(ValueError, match="sample-domain.*variance"):
        decode_noisy_network_pusch(
            composition,
            replace(noisy, noise_variance_w_per_sample=variance),
            _FakePUSCHReceiver(),
            noise_variance_w_per_re=(0.04,),
        )


@pytest.mark.parametrize("maximum", [True, 0, 1.5])
def test_network_receiver_requires_positive_integer_tensor_limit(maximum):
    composition, noisy = _network_waveform()
    with pytest.raises(ValueError, match="max_tensor_elements"):
        decode_noisy_network_pusch(
            composition,
            noisy,
            _FakePUSCHReceiver(),
            noise_variance_w_per_re=(0.04,),
            max_tensor_elements=maximum,
        )


def test_network_receiver_rejects_unpinned_implicit_noise_conversion():
    composition, noisy = _network_waveform()
    with pytest.raises(ValueError, match="pinned Sionna PHY 2.2.0"):
        decode_noisy_network_pusch(composition, noisy, _FakePUSCHReceiver())


@pytest.mark.parametrize(
    "filter_taps",
    [None, torch.zeros((1, 1, 2), dtype=torch.complex64)],
)
def test_network_receiver_reports_empty_declared_fir_support(filter_taps):
    composition, noisy = _network_waveform()
    result = decode_noisy_network_pusch(
        replace(composition, link_filter_taps=filter_taps),
        noisy,
        _FakePUSCHReceiver(),
        noise_variance_w_per_re=(0.04,),
    )

    diagnostic = result.report["declared_fir_cp_diagnostic"]
    assert diagnostic["maximum_fir_memory_samples"] == 0
    assert diagnostic["maximum_composed_support_span_samples"] == 0
    first_tap = 0 if filter_taps is None else -1
    last_tap = first_tap
    assert diagnostic["per_receiver_link_support"] == [
        [{"first_nonzero_tap": first_tap, "last_nonzero_tap": last_tap, "memory_samples": 0}]
    ]


@pytest.mark.parametrize(
    ("noise", "maximum", "message"),
    [
        ((True,), 100, "non-Boolean"),
        ((0.0,), 100, "finite and positive"),
        ((0.04, 0.05), 100, r"real \[rx\] vector"),
        ((0.04,), 1, "exceed max_tensor_elements"),
    ],
)
def test_network_receiver_fails_closed_on_noise_profile_and_resource_limit(noise, maximum, message):
    composition, noisy = _network_waveform()
    receiver = _FakePUSCHReceiver()
    with pytest.raises(ValueError, match=message):
        decode_noisy_network_pusch(
            composition,
            noisy,
            receiver,
            noise_variance_w_per_re=noise,
            max_tensor_elements=maximum,
        )
    assert not receiver.decoded


def test_network_receiver_requires_matching_ofdm_sample_rate_before_demodulation():
    composition, noisy = _network_waveform()
    receiver = _FakePUSCHReceiver()
    invalid_composition = replace(composition, sample_rate_hz=3.0)
    with pytest.raises(ValueError, match="sample rate does not match"):
        decode_noisy_network_pusch(
            invalid_composition,
            noisy,
            receiver,
            noise_variance_w_per_re=(0.04,),
        )
    assert not receiver.decoded


@pytest.mark.parametrize("sample_count", [7, 9])
def test_network_receiver_rejects_incomplete_or_trailing_ofdm_slot_samples(sample_count):
    links = torch.ones((1, 1, 1, 1, sample_count), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=sample_count,
        sample_rate_hz=4.0,
    )
    noisy = add_independent_receiver_waveform_awgn(composition, (0.02,), seed=19)
    receiver = _FakePUSCHReceiver()

    with pytest.raises(ValueError, match="sample count must exactly match"):
        decode_noisy_network_pusch(
            composition,
            noisy,
            receiver,
            noise_variance_w_per_re=(0.04,),
        )
    assert not receiver.decoded


@pytest.mark.parametrize("fft_size", [1, 24, 256, 4096])
def test_unitary_sionna_fft_preserves_iid_sample_noise_variance(fft_size):
    converted = _unitary_sionna_noise_conversion(
        torch,
        torch.tensor((0.0, 0.25), dtype=torch.float64),
        rx_count=2,
        fft_size=fft_size,
        device="cpu",
    )
    torch.testing.assert_close(converted, torch.tensor((0.0, 0.25), dtype=torch.float64))


@pytest.mark.parametrize(
    ("variance", "fft_size"),
    [
        ([0.25], 4),
        (torch.tensor([True]), 4),
        (torch.tensor([1 + 0j]), 4),
        (torch.tensor([1]), 4),
        (torch.tensor([float("inf")]), 4),
        (torch.tensor([-0.1]), 4),
        (torch.tensor([0.1]), True),
        (torch.tensor([0.1]), 0),
    ],
)
def test_unitary_sionna_noise_conversion_rejects_invalid_inputs(variance, fft_size):
    with pytest.raises(ValueError):
        _unitary_sionna_noise_conversion(
            torch,
            variance,
            rx_count=1,
            fft_size=fft_size,
            device="cpu",
        )


@pytest.mark.native_sionna
def test_native_network_waveform_decodes_on_selected_pusch_device():
    pytest.importorskip("sionna.phy")
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from receivers.nr import NRPUSCHConfig

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device=device,
    )
    tx_grid, _tx_bits = receiver.transmit(batch_size=1, seed=12_345)
    tx_waveform = receiver.modulate_slot_grid(tx_grid)
    sample_variance = 2e-4
    noise_generator = torch.Generator(device="cpu").manual_seed(54_319)
    noise_only_cpu = (
        torch.complex(
            torch.randn((512, 1, 1, tx_waveform.shape[-1]), generator=noise_generator),
            torch.randn((512, 1, 1, tx_waveform.shape[-1]), generator=noise_generator),
        )
        * (sample_variance / 2.0) ** 0.5
    )
    noise_only_grid = receiver.demodulate_slot_waveform(noise_only_cpu.to(device))
    measured_noise_variance = noise_only_grid.abs().square().mean()
    assert float(measured_noise_variance) == pytest.approx(sample_variance, rel=0.015)

    links = tx_waveform[:, None, :, :1, :]
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=tx_waveform.shape[-1],
        sample_rate_hz=receiver.resource_grid.fft_size * 30_000,
    )
    noisy = add_independent_receiver_waveform_awgn(
        composition,
        (1e-7,),
        seed=54_321,
    )

    result = decode_noisy_network_pusch(
        composition,
        noisy,
        receiver,
        noise_variance_w_per_re=None,
    )

    assert result.report["execution_device"] == device
    assert result.report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert result.report["shape"]["transport_blocks"] == 1
    torch.testing.assert_close(
        result.noise_variance_w_per_re,
        noisy.noise_variance_w_per_sample.to(result.noise_variance_w_per_re.device),
        rtol=0,
        atol=0,
    )
    assert result.report["noise_variance_conversion"] == (
        "sionna_phy_2_2_0_unitary_dft_preserves_variance_v1"
    )
    assert bool(result.crc_ok.all())

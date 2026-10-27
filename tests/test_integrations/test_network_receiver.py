import hashlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.network_phy import (  # noqa: E402
    add_independent_receiver_waveform_awgn,
    compose_received_waveforms,
)
from integrations.sionna.network_receiver import (  # noqa: E402
    _unitary_sionna_noise_conversion,
    decode_noisy_network_pusch,
)


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

    def demodulate_slot_waveform(self, waveform):
        assert waveform.shape == (1, 1, 1, 8)
        return torch.ones((1, 1, 1, 2, 4), dtype=torch.complex64, device=waveform.device)

    def decode_pusch(self, received_grid, noise_variance):
        self.decoded = True
        self.noise_variance = noise_variance.clone()
        return torch.zeros((1, 1, 12), dtype=torch.bool), torch.tensor([[True]])


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
    assert result.report["schema"] == "umcd.network_pusch_receiver_decode.v2"
    assert result.report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert result.report["receiver_noise_convention"] == "watts_per_resource_element"
    assert result.report["noise_variance_conversion"] == "caller_supplied_per_re_variance_v1"
    assert result.report["shape"]["transport_blocks"] == 1
    assert len(result.report["report_sha256"]) == 64
    report_body = dict(result.report)
    report_sha256 = report_body.pop("report_sha256")
    assert (
        report_sha256
        == hashlib.sha256(
            json.dumps(report_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


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

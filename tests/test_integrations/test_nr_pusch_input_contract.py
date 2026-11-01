"""Dependency-isolated input contracts for the NR PUSCH receiver adapter."""

from __future__ import annotations

import pytest
from receivers.nr import NRPUSCHConfig


def test_pusch_frequency_hopping_defaults_to_explicit_backend_supported_mode() -> None:
    config = NRPUSCHConfig()

    assert config.frequency_hopping == "neither"
    assert config.transform_precoding is False


@pytest.mark.parametrize("mode", ("intraSlot", "interSlot", "intra-slot", "unknown", True))
def test_pusch_frequency_hopping_rejects_unimplemented_modes(mode: object) -> None:
    with pytest.raises(ValueError, match="frequency_hopping currently supports only"):
        NRPUSCHConfig(frequency_hopping=mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, 1, "true", None))
def test_pusch_transform_precoding_fails_closed(value: object) -> None:
    if value is True:
        with pytest.raises(ValueError, match="does not implement DFT-s-OFDM PUSCH"):
            NRPUSCHConfig(transform_precoding=value)
    else:
        with pytest.raises(TypeError, match="transform_precoding must be a Boolean"):
            NRPUSCHConfig(transform_precoding=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "Boolean or complex"),
        (1.0 + 0.5j, "Boolean or complex"),
        (-0.1, "finite and nonnegative"),
        (float("nan"), "finite and nonnegative"),
        ("noise", "real numeric values"),
    ],
)
def test_pusch_noise_variance_rejects_nonphysical_values(value: object, message: str) -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    with pytest.raises(ValueError, match=message):
        _validated_noise_variance(torch, value, device="cpu")


def test_pusch_noise_variance_rejects_float32_overflow() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    value = torch.tensor(1e100, dtype=torch.float64)
    with pytest.raises(ValueError, match="representable as finite float32"):
        _validated_noise_variance(torch, value, device="cpu")


def test_pusch_noise_variance_rejects_positive_float32_underflow() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    value = torch.tensor(1e-100, dtype=torch.float64)
    with pytest.raises(ValueError, match="underflow to zero"):
        _validated_noise_variance(torch, value, device="cpu")


def test_pusch_noise_variance_rejects_python_scalar_underflow() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    with pytest.raises(ValueError, match="underflow to zero"):
        _validated_noise_variance(torch, 1e-100, device="cpu")


def test_pusch_noise_variance_accepts_nonnegative_real_tensors() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    value = torch.tensor([0, 2], dtype=torch.int32)
    result = _validated_noise_variance(torch, value, device="cpu")

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0]))


def _adapter_with_recording_decoder(torch, expected_tail=(14, 48)):
    from types import SimpleNamespace

    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver

    class RecordingDecoder:
        calls = 0

        def __call__(self, received_grid, noise_variance):
            self.calls += 1
            self.received_grid = received_grid
            self.noise_variance = noise_variance
            return "decoded", "crc"

    receiver = object.__new__(SionnaPUSCHLSReceiver)
    receiver._torch = torch
    receiver.device = "cpu"
    receiver._resource_grid = SimpleNamespace(
        num_ofdm_symbols=expected_tail[0], fft_size=expected_tail[1]
    )
    receiver._decoder = RecordingDecoder()
    return receiver, receiver._decoder


def test_pusch_decoder_validates_and_normalizes_grid_and_noise_before_backend_call() -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    grid = torch.ones((2, 1, 2, 14, 48), dtype=torch.complex128)

    result = receiver.decode_pusch(grid, torch.tensor([0.1, 0.2], dtype=torch.float64))

    assert result == ("decoded", "crc")
    assert decoder.calls == 1
    assert decoder.received_grid.dtype == torch.complex64
    assert decoder.received_grid.device.type == "cpu"
    assert tuple(decoder.noise_variance.shape) == (2, 1, 1)
    assert decoder.noise_variance.dtype == torch.float32


@pytest.mark.parametrize(
    ("grid_factory", "noise", "message"),
    [
        (lambda torch: torch.ones((1, 1, 1, 14, 48)), 0.1, "dense complex"),
        (
            lambda torch: torch.sparse_coo_tensor(
                torch.tensor([[0], [0], [0], [0], [0]]),
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
                (1, 1, 1, 14, 48),
                check_invariants=True,
            ),
            0.1,
            "dense complex",
        ),
        (
            lambda torch: torch.ones((1, 1, 14, 48), dtype=torch.complex64),
            0.1,
            "symbols, 48 subcarriers",
        ),
        (
            lambda torch: torch.ones((0, 1, 1, 14, 48), dtype=torch.complex64),
            0.1,
            "dimensions must be positive",
        ),
        (
            lambda torch: torch.full((1, 1, 1, 14, 48), complex(float("nan"), 0)),
            0.1,
            "received_grid must be finite",
        ),
        (
            lambda torch: torch.full((1, 1, 1, 14, 48), 1e300, dtype=torch.complex128),
            0.1,
            "remain finite when converted",
        ),
        (
            lambda torch: torch.ones((2, 1, 1, 14, 48), dtype=torch.complex64),
            [0.1, 0.2, 0.3],
            "must broadcast over",
        ),
        (
            lambda torch: torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64),
            True,
            "Boolean or complex",
        ),
    ],
)
def test_pusch_decoder_rejects_invalid_inputs_before_backend_call(
    grid_factory, noise, message
) -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)

    with pytest.raises(ValueError, match=message):
        receiver.decode_pusch(grid_factory(torch), noise)

    assert decoder.calls == 0

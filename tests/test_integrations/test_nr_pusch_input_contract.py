"""Dependency-isolated input contracts for the NR PUSCH receiver adapter."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from types import ModuleType, SimpleNamespace

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


def test_pusch_noise_variance_rejects_quantized_tensors_and_fallback_conversion_errors() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    unsupported_integer = torch.tensor([1], dtype=torch.uint16)
    with pytest.raises(ValueError, match="real numeric values"):
        _validated_noise_variance(torch, unsupported_integer, device="cpu")

    class ConversionFailureProxy:
        calls = 0

        def __getattr__(self, name):
            return getattr(torch, name)

        def as_tensor(self, value, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return torch.tensor(1)
            raise RuntimeError("fixture conversion failure")

    with pytest.raises(ValueError, match="real numeric values"):
        _validated_noise_variance(ConversionFailureProxy(), object(), device="cpu")


def _adapter_with_recording_decoder(torch, expected_tail=(14, 48), outputs=None):
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver

    class RecordingDecoder:
        calls = 0

        def __init__(self, outputs):
            self.outputs = outputs

        def __call__(self, received_grid, noise_variance, **kwargs):
            self.calls += 1
            self.received_grid = received_grid
            self.noise_variance = noise_variance
            self.channel_truth = kwargs.get("h")
            return self.outputs

    receiver = object.__new__(SionnaPUSCHLSReceiver)
    receiver._torch = torch
    receiver.device = "cpu"
    receiver._resource_grid = SimpleNamespace(
        num_ofdm_symbols=expected_tail[0],
        fft_size=expected_tail[1],
        cyclic_prefix_length=4,
        effective_subcarrier_ind=torch.arange(expected_tail[1]),
    )
    receiver.config = NRPUSCHConfig()
    receiver._metadata = {"receiver": "software-fixture"}
    if outputs is None:
        outputs = (
            torch.zeros((2, 1, 4), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.bool),
        )
    receiver._decoder = RecordingDecoder(outputs)
    return receiver, receiver._decoder


def _install_fake_sionna_phy(monkeypatch, torch, *, frequency_hopping="neither"):
    phy = ModuleType("sionna.phy")
    phy.config = SimpleNamespace(device="cpu")
    nr = ModuleType("sionna.phy.nr")

    def pusch_config():
        return SimpleNamespace(
            carrier=SimpleNamespace(),
            dmrs=SimpleNamespace(),
            frequency_hopping=frequency_hopping,
        )

    class Transmitter:
        def __init__(self, config, **_kwargs):
            self.config = config
            self.resource_grid = SimpleNamespace(
                pilot_pattern=SimpleNamespace(
                    mask=torch.ones((1, 1, 14, 48), dtype=torch.bool)
                ),
                num_ofdm_symbols=14,
                fft_size=48,
                cyclic_prefix_length=4,
                effective_subcarrier_ind=torch.arange(48),
            )

        def __call__(self, batch_size):
            return torch.zeros((batch_size, 1, 1, 14, 48)), torch.zeros(
                (batch_size, 1), dtype=torch.int32
            )

    class Estimator:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, grid, noise):
            return grid.unsqueeze(3).unsqueeze(4), noise[..., None, None, None, None]

    nr.PUSCHConfig = pusch_config
    nr.PUSCHTransmitter = Transmitter
    nr.PUSCHLSChannelEstimator = Estimator
    sionna = ModuleType("sionna")
    sionna.__path__ = []
    phy.__path__ = []
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.phy", phy)
    monkeypatch.setitem(sys.modules, "sionna.phy.nr", nr)
    return phy


def test_pusch_receiver_constructs_against_the_declared_sionna_contract(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import integrations.sionna.nr_pusch as adapter_module

    phy = _install_fake_sionna_phy(monkeypatch, torch)
    monkeypatch.setattr(adapter_module, "sionna_phy_version", lambda: "fixture-version")

    receiver = adapter_module.SionnaPUSCHLSReceiver(NRPUSCHConfig())

    assert receiver.device == "cpu"
    assert phy.config.device == "cpu"
    assert receiver.metadata["sionna_version"] == "fixture-version"
    assert receiver.metadata["receiver"] == "PUSCHLSChannelEstimator"
    assert tuple(receiver.pilot_mask.shape) == (1, 1, 14, 48)


def test_pusch_receiver_rejects_sionna_frequency_hopping_drift(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import integrations.sionna.nr_pusch as adapter_module

    _install_fake_sionna_phy(monkeypatch, torch, frequency_hopping="intraSlot")
    with pytest.raises(RuntimeError, match="frequency-hopping mode differs"):
        adapter_module.SionnaPUSCHLSReceiver(NRPUSCHConfig())


def test_pusch_receiver_rejects_unavailable_cuda_before_backend_construction(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import integrations.sionna.nr_pusch as adapter_module

    _install_fake_sionna_phy(monkeypatch, torch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device requested but unavailable"):
        adapter_module.SionnaPUSCHLSReceiver(NRPUSCHConfig(), device="cuda:0")


def test_pusch_receiver_rejects_unsupported_device_after_optional_import(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import integrations.sionna.nr_pusch as adapter_module

    _install_fake_sionna_phy(monkeypatch, torch)
    with pytest.raises(ValueError, match="device must be cpu or a CUDA device"):
        adapter_module.SionnaPUSCHLSReceiver(NRPUSCHConfig(), device="tpu")


def test_pusch_receiver_reports_missing_optional_dependency(monkeypatch) -> None:
    import builtins

    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver

    original_import = builtins.__import__

    def fail_sionna_import(name, *args, **kwargs):
        if name == "sionna.phy":
            raise ImportError("fixture dependency missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_sionna_import)
    with pytest.raises(RuntimeError, match="Install the optional Sionna 2.2 PHY dependencies"):
        SionnaPUSCHLSReceiver(NRPUSCHConfig())


def test_pusch_receiver_records_unknown_sionna_version(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import integrations.sionna.nr_pusch as adapter_module

    _install_fake_sionna_phy(monkeypatch, torch)
    monkeypatch.setattr(
        adapter_module,
        "sionna_phy_version",
        lambda: (_ for _ in ()).throw(PackageNotFoundError("sionna-phy")),
    )
    receiver = adapter_module.SionnaPUSCHLSReceiver(NRPUSCHConfig())
    assert receiver.metadata["sionna_version"] == "unknown"


@pytest.mark.parametrize(
    "method_name",
    ["decode_pusch", "make_decoder_with_estimator", "decode_pusch_with_channel_truth"],
)
def test_pusch_decoder_constructors_report_missing_optional_dependency(
    monkeypatch, method_name
) -> None:
    import builtins

    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    receiver._decoder = None
    receiver._perfect_csi_decoder = None
    original_import = builtins.__import__

    def fail_nr_import(name, *args, **kwargs):
        if name == "sionna.phy.nr":
            raise ImportError("fixture dependency missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_nr_import)
    with pytest.raises(RuntimeError, match="Install the optional Sionna 2.2 PHY dependencies"):
        if method_name == "decode_pusch":
            receiver.decode_pusch(torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64), 0.1)
        elif method_name == "make_decoder_with_estimator":
            receiver.make_decoder_with_estimator(object())
        else:
            receiver.decode_pusch_with_channel_truth(
                torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64),
                0.1,
                torch.ones((1, 1, 1, 1, 1, 14, 48), dtype=torch.complex64),
            )


def test_make_decoder_with_estimator_passes_the_supplied_estimator(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    receiver._transmitter = object()
    phy = ModuleType("sionna.phy")
    nr = ModuleType("sionna.phy.nr")
    calls = []
    nr.PUSCHReceiver = lambda *args, **kwargs: calls.append((args, kwargs)) or "decoder"
    phy.__path__ = []
    monkeypatch.setitem(sys.modules, "sionna.phy", phy)
    monkeypatch.setitem(sys.modules, "sionna.phy.nr", nr)
    estimator = object()

    result = receiver.make_decoder_with_estimator(estimator)

    assert result == "decoder"
    assert calls[0][1]["channel_estimator"] is estimator
    assert calls[0][1]["return_tb_crc_status"] is True


@pytest.mark.parametrize("batch_size", [True, 0, 1.5])
def test_pusch_transmitter_rejects_invalid_batch_size(batch_size) -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    receiver._transmitter = lambda _batch_size: None
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        receiver.transmit(batch_size)


def test_pusch_transmitter_uses_configured_seed_boundary(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import integrations.sionna.nr_pusch as adapter_module

    receiver, _ = _adapter_with_recording_decoder(torch)
    receiver._transmitter = lambda batch_size: ("grid", batch_size)
    calls = []

    def seeded(device, seed, operation):
        calls.append((device, seed))
        return operation()

    monkeypatch.setattr(adapter_module, "call_with_sionna_seed", seeded)
    assert receiver.transmit(3, seed=17) == ("grid", 3)
    assert calls == [("cpu", 17)]


@pytest.mark.parametrize(
    ("grid_factory", "noise_factory", "message"),
    [
        (lambda torch: object(), 0.1, "dense complex"),
        (
            lambda torch: torch.ones((1, 1, 0, 14, 48), dtype=torch.complex64),
            0.1,
            "dimensions must be positive",
        ),
        (
            lambda torch: torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64),
            lambda torch: torch.ones((1, 1, 1, 1)),
            "must broadcast over",
        ),
        (
            lambda torch: torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64),
            lambda torch: torch.sparse_coo_tensor(
                torch.tensor([[0]]), torch.tensor([0.1]), (1,), check_invariants=True
            ),
            "dense real-valued",
        ),
    ],
)
def test_pusch_decoder_rejects_unconvertible_grid_and_nonbroadcastable_noise(
    grid_factory, noise_factory, message
) -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    noise = noise_factory(torch) if callable(noise_factory) else noise_factory
    with pytest.raises(ValueError, match=message):
        receiver.decode_pusch(grid_factory(torch), noise)
    assert decoder.calls == 0


def test_pusch_allocation_grid_places_and_extracts_configured_slot_symbols() -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch, expected_tail=(4, 48))
    receiver.config = NRPUSCHConfig(mapping_type="B", symbol_allocation=(2, 4))
    allocation = torch.arange(4 * 48, dtype=torch.float32).to(torch.complex64).reshape(
        1, 1, 1, 4, 48
    )

    slot = receiver.place_allocation_in_slot(allocation)
    extracted = receiver.extract_allocation_from_slot(slot)

    assert tuple(slot.shape) == (1, 1, 1, 14, 48)
    assert torch.count_nonzero(slot[..., :2, :]) == 0
    assert torch.count_nonzero(slot[..., 6:, :]) == 0
    torch.testing.assert_close(extracted, allocation)


@pytest.mark.parametrize(
    ("method_name", "grid", "message"),
    [
        (
            "place_allocation_in_slot",
            lambda torch: torch.ones((1, 1, 1, 13, 48), dtype=torch.complex64),
            "symbol/FFT axes",
        ),
        (
            "place_allocation_in_slot",
            lambda torch: torch.full(
                (1, 1, 1, 14, 48), complex(float("nan"), 0), dtype=torch.complex64
            ),
            "allocation_grid must be finite",
        ),
        (
            "extract_allocation_from_slot",
            lambda torch: torch.ones((1, 1, 1, 14, 48)),
            "complex dtype",
        ),
        (
            "extract_allocation_from_slot",
            lambda torch: torch.ones((1, 1, 1, 14, 48, 1), dtype=torch.complex64),
            "axes",
        ),
    ],
)
def test_pusch_slot_helpers_reject_grid_axes_and_finiteness(method_name, grid, message) -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    with pytest.raises(ValueError, match=message):
        getattr(receiver, method_name)(grid(torch))


def test_pusch_bwp_grid_maps_to_carrier_crb_bins_and_roundtrips() -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch, expected_tail=(4, 48))
    receiver.config = NRPUSCHConfig(
        n_size_bwp=4,
        n_size_grid=10,
        n_start_grid=1,
        n_start_bwp=3,
        mapping_type="B",
        symbol_allocation=(2, 4),
    )
    allocation = torch.ones((1, 1, 1, 4, 48), dtype=torch.complex64)

    carrier_slot = receiver.place_bwp_allocation_in_carrier_slot(allocation)
    restored = receiver.extract_bwp_allocation_from_carrier_slot(carrier_slot)

    assert tuple(carrier_slot.shape) == (1, 1, 1, 14, 120)
    torch.testing.assert_close(carrier_slot[..., 2:6, 24:72], allocation)
    torch.testing.assert_close(restored, allocation)


@pytest.mark.parametrize(
    ("method_name", "value", "message"),
    [
        ("place_allocation_in_slot", lambda torch: torch.ones((1, 1, 1, 4, 48)), "complex dtype"),
        (
            "place_allocation_in_slot",
            lambda torch: torch.ones((1, 1, 4, 48), dtype=torch.complex64),
            "axes",
        ),
        (
            "extract_allocation_from_slot",
            lambda torch: torch.ones((1, 1, 1, 13, 48), dtype=torch.complex64),
            "end with",
        ),
        (
            "extract_allocation_from_slot",
            lambda torch: torch.full(
                (1, 1, 1, 14, 48), complex(float("nan"), 0), dtype=torch.complex64
            ),
            "finite",
        ),
    ],
)
def test_pusch_slot_allocation_helpers_reject_invalid_grids(method_name, value, message) -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    with pytest.raises(ValueError, match=message):
        getattr(receiver, method_name)(value(torch))


def _install_fake_ofdm(monkeypatch):
    ofdm = ModuleType("sionna.phy.ofdm")
    ofdm.calls = []

    class Transform:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            ofdm.calls.append(kwargs)

        def __call__(self, values):
            return values

    ofdm.OFDMModulator = Transform
    ofdm.OFDMDemodulator = Transform
    monkeypatch.setitem(sys.modules, "sionna.phy.ofdm", ofdm)
    return ofdm


def test_pusch_ofdm_modulators_and_demodulators_use_receiver_parameters(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    receiver, _ = _adapter_with_recording_decoder(torch)
    ofdm = _install_fake_ofdm(monkeypatch)
    slot_grid = torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64)
    carrier_waveform = torch.ones((1, 1, 1, 14 * (48 + 4)), dtype=torch.complex64)

    assert receiver.modulate_slot_grid(slot_grid) is slot_grid
    assert receiver.demodulate_slot_waveform(carrier_waveform) is carrier_waveform
    assert receiver.modulate_carrier_slot_grid(slot_grid) is slot_grid
    assert receiver.demodulate_carrier_slot_waveform(carrier_waveform) is carrier_waveform
    assert [call["fft_size"] for call in ofdm.calls if "fft_size" in call] == [48, 48]
    assert [call["cyclic_prefix_length"] for call in ofdm.calls] == [4, 4, 4, 4]


@pytest.mark.parametrize(
    ("method_name", "argument"),
    [
        ("modulate_carrier_slot_grid", lambda torch: torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64)),
        ("modulate_slot_grid", lambda torch: torch.ones((1, 1, 1, 14, 48), dtype=torch.complex64)),
        ("demodulate_slot_waveform", lambda torch: torch.ones((1, 1, 1, 1), dtype=torch.complex64)),
        ("demodulate_carrier_slot_waveform", lambda torch: torch.ones((1, 1, 1, 728), dtype=torch.complex64)),
    ],
)
def test_pusch_ofdm_operations_report_missing_optional_dependency(
    monkeypatch, method_name, argument
) -> None:
    import builtins

    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    original_import = builtins.__import__

    def fail_ofdm_import(name, *args, **kwargs):
        if name == "sionna.phy.ofdm":
            raise ImportError("fixture dependency missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_ofdm_import)
    with pytest.raises(RuntimeError, match="Install the optional Sionna 2.2 PHY dependencies"):
        getattr(receiver, method_name)(argument(torch))


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (lambda torch: torch.ones((1, 1, 1, 728)), "dense complex"),
        (
            lambda torch: torch.ones((1, 1, 1, 727), dtype=torch.complex64),
            "exactly one configured full-carrier slot",
        ),
        (
            lambda torch: torch.full(
                (1, 1, 1, 728), complex(float("nan"), 0), dtype=torch.complex64
            ),
            "waveform must be finite",
        ),
    ],
)
def test_carrier_waveform_demodulator_rejects_invalid_shape_dtype_or_values(
    monkeypatch, waveform, message
) -> None:
    torch = pytest.importorskip("torch")

    receiver, _ = _adapter_with_recording_decoder(torch)
    _install_fake_ofdm(monkeypatch)
    with pytest.raises(ValueError, match=message):
        receiver.demodulate_carrier_slot_waveform(waveform(torch))


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (lambda torch: torch.ones((1, 1, 728), dtype=torch.complex64), "waveform must have"),
        (
            lambda torch: torch.ones((1, 0, 1, 728), dtype=torch.complex64),
            "waveform must have",
        ),
    ],
)
def test_carrier_waveform_demodulator_rejects_invalid_axes(
    monkeypatch, waveform, message
) -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch)
    _install_fake_ofdm(monkeypatch)
    with pytest.raises(ValueError, match=message):
        receiver.demodulate_carrier_slot_waveform(waveform(torch))


@pytest.mark.parametrize(
    ("carrier_grid", "message"),
    [
        (lambda torch: torch.ones((1, 1, 1, 14, 48)), "complex dtype"),
        (
            lambda torch: torch.ones((1, 1, 1, 13, 48), dtype=torch.complex64),
            "carrier FFT bins",
        ),
        (
            lambda torch: torch.full(
                (1, 1, 1, 14, 48), complex(float("nan"), 0), dtype=torch.complex64
            ),
            "carrier_slot_grid must be finite",
        ),
    ],
)
def test_bwp_extractor_rejects_invalid_carrier_grid(carrier_grid, message) -> None:
    torch = pytest.importorskip("torch")

    receiver, _ = _adapter_with_recording_decoder(torch)
    with pytest.raises(ValueError, match=message):
        receiver.extract_bwp_allocation_from_carrier_slot(carrier_grid(torch))


def test_pusch_decoder_validates_and_normalizes_grid_and_noise_before_backend_call() -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    grid = torch.ones((2, 1, 2, 14, 48), dtype=torch.complex128)

    result = receiver.decode_pusch(grid, torch.tensor([0.1, 0.2], dtype=torch.float64))

    assert tuple(result[0].shape) == (2, 1, 4)
    assert tuple(result[1].shape) == (2, 1)
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


@pytest.mark.parametrize(
    ("bits", "crc", "message"),
    [
        (
            lambda torch: torch.tensor([[[0.0, 0.5]]]),
            lambda torch: torch.ones((1, 1), dtype=torch.bool),
            "binary",
        ),
        (
            lambda torch: torch.ones((1, 1, 2), dtype=torch.complex64),
            lambda torch: torch.ones((1, 1), dtype=torch.bool),
            "binary",
        ),
        (
            lambda torch: torch.ones((1, 1, 2), dtype=torch.float32),
            lambda torch: torch.ones((1, 1)),
            "binary",
        ),
        (
            lambda torch: torch.zeros((1, 1, 2), dtype=torch.float32),
            lambda torch: torch.ones((1, 1), dtype=torch.bool),
            "batch-aligned",
        ),
        (
            lambda torch: torch.zeros((2, 1, 2), dtype=torch.float32),
            lambda torch: torch.ones((2, 2), dtype=torch.bool),
            "batch-aligned",
        ),
        (
            lambda torch: torch.sparse_coo_tensor(
                torch.tensor([[0], [0], [0]]), torch.tensor([1.0]), (2, 1, 2), check_invariants=True
            ),
            lambda torch: torch.ones((2, 1), dtype=torch.bool),
            "batch-aligned",
        ),
    ],
)
def test_pusch_decoder_rejects_malformed_backend_outputs(bits, crc, message) -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch, outputs=(bits(torch), crc(torch)))

    with pytest.raises(ValueError, match=message):
        receiver.decode_pusch(torch.ones((2, 1, 1, 14, 48), dtype=torch.complex64), 0.1)

    assert decoder.calls == 1


def test_pusch_decoder_rejects_wrong_result_arity() -> None:
    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(torch, outputs=("only-one-output",))
    with pytest.raises(ValueError, match="decoded bits and CRC status"):
        receiver.decode_pusch(
            torch.ones((2, 1, 1, 14, 48), dtype=torch.complex64), 0.1
        )


def test_perfect_csi_decoder_validates_channel_truth_and_binary_outputs() -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    receiver._perfect_csi_decoder = decoder
    grid = torch.ones((2, 1, 1, 14, 48), dtype=torch.complex64)
    channel_truth = torch.ones((2, 1, 1, 1, 1, 14, 48), dtype=torch.complex128)

    bits, crc = receiver.decode_pusch_with_channel_truth(grid, 0.1, channel_truth)

    assert tuple(bits.shape) == (2, 1, 4)
    assert tuple(crc.shape) == (2, 1)
    assert decoder.calls == 1
    assert decoder.channel_truth.dtype == torch.complex64


@pytest.mark.parametrize(
    ("channel_factory", "message"),
    [
        (lambda torch: torch.ones((2, 1, 14, 48), dtype=torch.complex64), "axes matching"),
        (
            lambda torch: torch.ones((1, 1, 1, 1, 1, 14, 48), dtype=torch.complex64),
            "axes matching",
        ),
        (
            lambda torch: torch.full((2, 1, 1, 1, 1, 14, 48), complex(float("nan"), 0)),
            "channel_truth must be finite",
        ),
    ],
)
def test_perfect_csi_decoder_rejects_malformed_channel_truth_before_backend_call(
    channel_factory, message
) -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    receiver._perfect_csi_decoder = decoder

    with pytest.raises(ValueError, match=message):
        receiver.decode_pusch_with_channel_truth(
            torch.ones((2, 1, 1, 14, 48), dtype=torch.complex64),
            0.1,
            channel_factory(torch),
        )

    assert decoder.calls == 0


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (lambda torch: object(), "channel_truth must be a dense finite complex tensor"),
        (
            lambda torch: torch.full(
                (2, 1, 1, 1, 1, 14, 48), 1e300 + 0j, dtype=torch.complex128
            ),
            "remain finite when converted",
        ),
    ],
)
def test_perfect_csi_decoder_rejects_unconvertible_or_unrepresentable_truth(
    truth, message
) -> None:
    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    receiver._perfect_csi_decoder = decoder
    with pytest.raises(ValueError, match=message):
        receiver.decode_pusch_with_channel_truth(
            torch.ones((2, 1, 1, 14, 48), dtype=torch.complex64), 0.1, truth(torch)
        )
    assert decoder.calls == 0


def _matching_hop_metadata(config):
    return {
        "carrier_grid_start_rb": config.n_start_grid,
        "carrier_grid_size_rb": config.carrier_grid_size,
        "first_hop_start_crb": config.n_start_bwp,
        "allocation_rb_count": config.n_size_bwp,
        "allocation_start_symbol": config.symbol_allocation[0],
        "allocation_symbol_count": config.symbol_allocation[1],
    }


def test_pusch_dehopped_decode_records_unqualified_replay(monkeypatch) -> None:
    import numpy as np
    import receivers.nr.pusch as nr_pusch

    torch = pytest.importorskip("torch")
    receiver, _ = _adapter_with_recording_decoder(
        torch,
        outputs=(torch.zeros((1, 1, 4)), torch.ones((1, 1), dtype=torch.bool)),
    )
    monkeypatch.setattr(
        nr_pusch,
        "dehop_pusch_intra_slot_received_grid",
        lambda _grid, _metadata: SimpleNamespace(
            frequency_grid=np.ones((1, 1, 1, 14, 48), dtype=np.complex64),
            metadata={"fixture": "dehopped"},
        ),
    )
    monkeypatch.setattr(
        nr_pusch,
        "verify_pusch_intra_slot_dehopped_grid",
        lambda *_args: {"verified": True},
    )

    bits, crc, report = receiver.decode_intra_slot_hopped_carrier_grid(
        np.ones((1, 1, 1, 14, 48), dtype=np.complex64),
        _matching_hop_metadata(receiver.config),
        0.1,
    )

    assert tuple(bits.shape) == (1, 1, 4)
    assert tuple(crc.shape) == (1, 1)
    assert report["status"] == "DECODED_NOT_QUALIFIED"
    assert report["receiver_hopping_qualification"] == "NOT_QUALIFIED"
    assert report["dehop_replay"] == {"verified": True}


@pytest.mark.parametrize(
    ("grid", "metadata", "message"),
    [
        (lambda np: [[[[[0j]]]]], lambda config: _matching_hop_metadata(config), "NumPy array"),
        (
            lambda np: np.ones((1, 1, 1, 14, 48), dtype=np.complex64),
            lambda config: None,
            "must be a mapping",
        ),
        (
            lambda np: np.ones((1, 1, 1, 14, 48), dtype=np.complex64),
            lambda config: {**_matching_hop_metadata(config), "carrier_grid_size_rb": 3},
            "carrier_grid_size_rb does not match",
        ),
    ],
)
def test_pusch_dehopped_decode_rejects_invalid_inputs_before_dehop(
    grid, metadata, message
) -> None:
    import numpy as np

    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    with pytest.raises((TypeError, ValueError), match=message):
        receiver.decode_intra_slot_hopped_carrier_grid(
            grid(np), metadata(receiver.config), 0.1
        )
    assert decoder.calls == 0


def test_pusch_dehopped_decode_rejects_receiver_with_frequency_hopping() -> None:
    import numpy as np

    torch = pytest.importorskip("torch")
    receiver, decoder = _adapter_with_recording_decoder(torch)
    object.__setattr__(receiver.config, "frequency_hopping", "intraSlot")
    with pytest.raises(ValueError, match="requires the supported non-hopping receiver"):
        receiver.decode_intra_slot_hopped_carrier_grid(
            np.ones((1, 1, 1, 14, 48), dtype=np.complex64),
            _matching_hop_metadata(receiver.config),
            0.1,
        )
    assert decoder.calls == 0

"""Pure input and frequency-grid tests for the RT-to-DMRS observation adapter."""

from __future__ import annotations

import builtins
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from integrations.sionna.rt_dmrs import (
    _label_semantics,
    build_rt_dmrs_generation_request,
    build_rt_dmrs_observation_dataset,
    read_stable_input_artifact,
    resample_centered_cfr,
    thermal_noise_variance_w,
)


def _labelled_dataset(*, labels, declared="unspecified_legacy"):
    count = len(labels)
    return ChannelDataset(
        h=np.ones((count, 1, 16), dtype=np.complex64),
        y=np.asarray([index % 2 for index in range(count)], dtype=np.uint8),
        group=np.asarray([f"g{index}" for index in range(count)]),
        domain=np.asarray(["scene"] * count),
        source=np.asarray(["sionna_rt"] * count),
        metadata=tuple({"geometry_label": label} for label in labels),
        label_semantics=declared,
    )


def test_rt_dmrs_label_semantics_resolves_legacy_metadata_and_declared_value():
    legacy = _labelled_dataset(
        labels=("GEOMETRIC_DIRECT_PATH_ORACLE", "GEOMETRIC_DIRECT_PATH_ORACLE")
    )
    declared = _labelled_dataset(
        labels=("GEOMETRIC_DIRECT_PATH_ORACLE", "GEOMETRIC_DIRECT_PATH_ORACLE"),
        declared="geometric_direct_path_oracle",
    )

    assert _label_semantics(legacy) == "geometric_direct_path_oracle"
    assert _label_semantics(declared) == "geometric_direct_path_oracle"


def test_rt_dmrs_label_semantics_rejects_mixed_or_unknown_targets():
    mixed = _labelled_dataset(labels=("GEOMETRIC_DIRECT_PATH_ORACLE", "PROFILE_PROXY"))
    unknown = _labelled_dataset(labels=("UNKNOWN", "UNKNOWN"))

    with pytest.raises(ValueError, match="consistent label semantic"):
        _label_semantics(mixed)
    with pytest.raises(ValueError, match="consistent label semantic"):
        _label_semantics(unknown)


def test_rt_dmrs_label_semantics_uses_declared_value_for_missing_metadata_and_rejects_conflicts():
    declared = replace(
        _labelled_dataset(labels=("PROFILE_PROXY",)),
        metadata=({},),
        label_semantics="profile_proxy",
    )
    assert _label_semantics(declared) == "profile_proxy"
    conflict = replace(
        _labelled_dataset(labels=("PROFILE_PROXY",)),
        label_semantics="geometric_direct_path_oracle",
    )
    with pytest.raises(ValueError, match="conflicts with its declared semantics"):
        _label_semantics(conflict)
    unresolved = replace(
        _labelled_dataset(labels=("PROFILE_PROXY",)),
        metadata=({},),
        label_semantics="unspecified_legacy",
    )
    with pytest.raises(ValueError, match="consistent label semantic"):
        _label_semantics(unresolved)


def test_centered_cfr_resampling_preserves_overlapping_subcarriers():
    source_frequencies = (np.arange(128) - 64) * 30_000.0
    channels = np.stack(
        [
            (1.0 + source_frequencies / 1e6) + 1j * (0.5 - source_frequencies / 2e6),
            (-0.25 + source_frequencies / 3e6) + 1j * (1.0 + source_frequencies / 4e6),
        ]
    )[:, None, :].astype(np.complex64)

    actual = resample_centered_cfr(
        channels,
        source_spacing_hz=30_000.0,
        target_subcarriers=72,
        target_spacing_hz=30_000.0,
    )

    np.testing.assert_array_equal(actual, channels[:, :, 28:100])


def test_centered_cfr_resampling_preserves_multiple_receive_antennas():
    frequencies = (np.arange(128) - 64) * 30_000.0
    channels = np.stack(
        [
            np.stack(
                [
                    np.exp(-2j * np.pi * frequencies * 20e-9),
                    0.5 * np.exp(-2j * np.pi * frequencies * 80e-9),
                ]
            )
        ]
    ).astype(np.complex64)

    actual = resample_centered_cfr(
        channels,
        source_spacing_hz=30_000.0,
        target_subcarriers=72,
        target_spacing_hz=30_000.0,
    )

    assert actual.shape == (1, 2, 72)
    np.testing.assert_array_equal(actual, channels[:, :, 28:100])


def test_thermal_noise_variance_uses_temperature_bandwidth_and_noise_figure():
    actual = thermal_noise_variance_w(
        temperature_k=290.0,
        noise_figure_db=7.0,
        bandwidth_hz=30_000.0,
    )
    expected = 1.380649e-23 * 290.0 * 30_000.0 * 10.0**0.7
    assert actual == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("temperature_k", "noise_figure_db", "bandwidth_hz"),
    [
        (0.0, 7.0, 30_000.0),
        (290.0, -1.0, 30_000.0),
        (290.0, 7.0, float("inf")),
    ],
)
def test_thermal_noise_variance_rejects_invalid_physical_inputs(
    temperature_k, noise_figure_db, bandwidth_hz
):
    with pytest.raises(ValueError):
        thermal_noise_variance_w(
            temperature_k=temperature_k,
            noise_figure_db=noise_figure_db,
            bandwidth_hz=bandwidth_hz,
        )


@pytest.mark.parametrize(
    ("channels", "overrides", "message"),
    [
        (np.ones((2, 0, 128), dtype=np.complex64), {}, "rx_ant"),
        (np.ones((2, 1, 128), dtype=np.float64), {}, "complex"),
        (np.ones((2, 1, 128), dtype=np.complex64), {"source_spacing_hz": 0.0}, "source_spacing_hz"),
        (
            np.ones((2, 1, 128), dtype=np.complex64),
            {"target_subcarriers": 128, "target_spacing_hz": 60_000.0},
            "inside the source CFR bandwidth",
        ),
    ],
)
def test_centered_cfr_resampling_rejects_invalid_grids(channels, overrides, message):
    arguments = {
        "source_spacing_hz": 30_000.0,
        "target_subcarriers": 72,
        "target_spacing_hz": 30_000.0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        resample_centered_cfr(channels, **arguments)


@pytest.mark.parametrize("target_subcarriers", [True, 7])
def test_centered_cfr_rejects_invalid_target_counts(target_subcarriers):
    with pytest.raises(ValueError, match="target_subcarriers must be an integer"):
        resample_centered_cfr(
            np.ones((1, 1, 16), dtype=np.complex64),
            source_spacing_hz=30_000.0,
            target_subcarriers=target_subcarriers,
            target_spacing_hz=30_000.0,
        )


def test_stable_input_artifact_returns_digest_and_file_identity(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"stable input")
    payload, identity = read_stable_input_artifact(path)
    assert payload == b"stable input"
    assert identity["sha256"] == __import__("hashlib").sha256(payload).hexdigest()
    assert identity["size"] == str(len(payload))
    assert identity["inode"] == str(path.stat().st_ino)


def test_stable_input_artifact_rejects_nonregular_unlinked_and_changed_inputs(
    tmp_path, monkeypatch
):
    from integrations.sionna import rt_dmrs

    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        read_stable_input_artifact(fifo)

    path = tmp_path / "input.bin"
    path.write_bytes(b"bytes")
    monkeypatch.setattr(rt_dmrs.os, "readlink", lambda _path: f"{path} (deleted)")
    with pytest.raises(ValueError, match="unlinked while open"):
        read_stable_input_artifact(path)
    monkeypatch.undo()

    class ChangedReader:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            os.close(self.descriptor)

        @staticmethod
        def read():
            return b"changed-size"

    monkeypatch.setattr(rt_dmrs.os, "fdopen", lambda descriptor, _mode: ChangedReader(descriptor))
    with pytest.raises(PermissionError, match="changed while being read"):
        read_stable_input_artifact(path)


def test_generation_request_binds_dataset_and_source_artifact_metadata(tmp_path):
    dataset = _labelled_dataset(labels=("PROFILE_PROXY", "PROFILE_PROXY"))
    artifact = {"path": str(tmp_path / "input.bin"), "sha256": "a" * 64}
    request = build_rt_dmrs_generation_request(
        [dataset],
        source_artifacts=[artifact],
        destination=tmp_path / "out.npz",
        seed=17,
        device="cpu",
    )
    assert request["operation"] == "rt_dmrs_observations"
    assert request["sample_count"] == request["total_work_units"] == 2
    assert request["input_artifact_count"] == 1
    assert request["input_artifacts"] == [artifact]
    assert request["receiver_parameters"]["seed"] == 17
    assert request["destinations"] == [str((tmp_path / "out.npz").resolve())]


def _grid_dataset(*, spacing_hz=30_000.0, n_subcarriers=16):
    source = _labelled_dataset(labels=("PROFILE_PROXY",))
    metadata = ({
        "geometry_label": "PROFILE_PROXY",
        "grid_config": {"spacing_hz": spacing_hz, "n_subcarriers": n_subcarriers},
    },)
    return replace(source, metadata=metadata)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"snr_db": 12.0, "tx_power_per_re_w": 1e-3}, "snr_db must be None"),
        (
            {"snr_db": None, "tx_power_per_re_w": 0.0},
            "tx_power_per_re_w must be finite and positive",
        ),
        ({"snr_db": None}, "snr_db must be finite"),
        ({"seed": True}, "seed must be an integer"),
        ({"n_size_bwp": 0}, "n_size_bwp must be a positive integer"),
        ({"subcarrier_spacing_khz": False}, "subcarrier_spacing_khz must be a positive integer"),
    ],
)
def test_observation_builder_rejects_invalid_receiver_parameters(
    tmp_path, monkeypatch, options, message
):
    from integrations.sionna import rt_dmrs

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match=message):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset()], destination=tmp_path / "out.npz", **options
        )


def test_observation_builder_checks_destination_dataset_and_authorization(tmp_path):
    with pytest.raises(PermissionError, match="bind a scratch output destination"):
        build_rt_dmrs_observation_dataset([_grid_dataset()])
    with pytest.raises(ValueError, match="at least one RT dataset"):
        build_rt_dmrs_observation_dataset([], destination=tmp_path / "out.npz")
    with pytest.raises(PermissionError, match="authorization"):
        build_rt_dmrs_observation_dataset([_grid_dataset()], destination=tmp_path / "out.npz")


def test_observation_builder_rejects_grid_disagreement_before_native_import(tmp_path, monkeypatch):
    from integrations.sionna import rt_dmrs

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="same frequency grid"):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset(), _grid_dataset(spacing_hz=15_000.0)],
            destination=tmp_path / "out.npz",
        )


def test_observation_builder_checks_each_sample_grid_metadata(tmp_path, monkeypatch):
    from integrations.sionna import rt_dmrs

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    source = _labelled_dataset(labels=("PROFILE_PROXY", "PROFILE_PROXY"))
    metadata = (
        {"geometry_label": "PROFILE_PROXY", "grid_config": {"spacing_hz": 30_000.0, "n_subcarriers": 16}},
        {"geometry_label": "PROFILE_PROXY", "grid_config": {"spacing_hz": 15_000.0, "n_subcarriers": 16}},
    )
    mismatched = replace(source, metadata=metadata)
    with pytest.raises(ValueError, match="same frequency grid"):
        build_rt_dmrs_observation_dataset(
            [mismatched], destination=tmp_path / "out.npz", device="cpu"
        )


def test_observation_builder_rejects_metadata_tone_count_mismatch(tmp_path, monkeypatch):
    from integrations.sionna import rt_dmrs

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="metadata must match the CFR tone count"):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset(n_subcarriers=8)],
            destination=tmp_path / "out.npz",
            device="cpu",
        )


def test_observation_builder_rejects_invalid_device_after_grid_validation(tmp_path, monkeypatch):
    from integrations.sionna import rt_dmrs

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="device must be cpu or a CUDA device"):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset()], destination=tmp_path / "out.npz", device="metal"
        )


def test_observation_builder_pilot_mode_checks_authorized_destination(tmp_path, monkeypatch):
    from integrations.sionna import d1_generation_gate, rt_dmrs

    authorization = SimpleNamespace(mode="pilot")
    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    observed = []
    monkeypatch.setattr(
        d1_generation_gate,
        "require_pilot_destination",
        lambda auth, output, *, publisher: observed.append((auth, output, publisher)),
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="Install PyTorch"):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset()],
            authorization=authorization,
            destination=tmp_path / "out.npz",
            device="cpu",
        )
    assert observed == [(authorization, str((tmp_path / "out.npz").resolve()), "dmrs_dataset")]


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_observation_builder_rejects_unavailable_cuda_after_grid_checks(
    tmp_path, monkeypatch, device
):
    from integrations.sionna import rt_dmrs

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return FakeTorch
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="CUDA device requested but unavailable"):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset()], destination=tmp_path / "out.npz", device=device
        )


def test_observation_builder_rejects_full_band_mismatch_and_zero_channel_power(
    tmp_path, monkeypatch
):
    from integrations.sionna import rt_dmrs

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)

    class Receiver:
        def __init__(self, *_args, **_kwargs):
            self.resource_grid = self.grid

    Receiver.grid = SimpleNamespace(num_effective_subcarriers=8, fft_size=16)

    monkeypatch.setattr(rt_dmrs, "SionnaPUSCHLSReceiver", Receiver)
    with pytest.raises(ValueError, match="full-band PUSCH resource grid"):
        build_rt_dmrs_observation_dataset(
            [_grid_dataset()], destination=tmp_path / "out.npz", device="cpu"
        )

    Receiver.grid = SimpleNamespace(num_effective_subcarriers=16, fft_size=16)
    monkeypatch.setattr(rt_dmrs, "SionnaPUSCHLSReceiver", lambda *_a, **_k: Receiver())
    zero = _grid_dataset()
    object.__setattr__(zero, "h", np.zeros_like(zero.h))
    with pytest.raises(ValueError, match="positive power"):
        build_rt_dmrs_observation_dataset(
            [zero], destination=tmp_path / "out.npz", device="cpu"
        )


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def expand(self, *shape):
        return _FakeTensor(np.broadcast_to(self.value, shape))

    def reshape(self, *shape):
        return _FakeTensor(self.value.reshape(shape))

    def __mul__(self, other):
        value = other.value if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.value * value)

    def __add__(self, other):
        value = other.value if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.value + value)

    def abs(self):
        return _FakeTensor(np.abs(self.value))

    def square(self):
        return _FakeTensor(np.square(self.value))

    def mean(self, dim=None):
        return _FakeTensor(np.mean(self.value, axis=dim))

    def __getitem__(self, key):
        return _FakeTensor(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_observation_builder_emits_estimated_channel_in_snr_and_physical_power_modes(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext

    from integrations.sionna import rt_dmrs

    class FakeTorch:
        complex64 = "complex64"
        float32 = "float32"

        @staticmethod
        def as_tensor(value, **_kwargs):
            return _FakeTensor(value)

        class Generator:
            def __init__(self, **_kwargs):
                pass

            def manual_seed(self, _seed):
                return self

        @staticmethod
        def no_grad():
            return nullcontext()

    class Receiver:
        resource_grid = SimpleNamespace(num_effective_subcarriers=16, fft_size=16)
        metadata = {"receiver": "fixture"}

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def transmit(_batch, *, seed):
            return _FakeTensor(np.ones((1, 1, 1, 1, 16), dtype=np.complex64)), seed

        @staticmethod
        def estimate(received, _variance):
            shape = (received.shape[0], 1, 1, 1, 1, 1, 16)
            values = np.ones(shape, dtype=np.complex64)
            return SimpleNamespace(h_hat=_FakeTensor(values))

    monkeypatch.setattr(rt_dmrs, "require_d1_authorization", lambda *_a, **_k: None)
    monkeypatch.setattr(rt_dmrs, "SionnaPUSCHLSReceiver", Receiver)
    monkeypatch.setattr(
        rt_dmrs,
        "proper_complex_awgn",
        lambda signal, *_a, **_k: _FakeTensor(np.zeros_like(signal.value)),
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    normalized = build_rt_dmrs_observation_dataset(
        [_grid_dataset()], destination=tmp_path / "normalized.npz", device="cpu", seed=5
    )
    assert normalized.h.shape == (1, 1, 16)
    assert normalized.generation_report["power_mode"] == "normalized_per_link_snr"
    assert normalized.metadata[0]["receiver_observation"].startswith("Sionna PUSCH DMRS")

    physical = build_rt_dmrs_observation_dataset(
        [_grid_dataset()],
        destination=tmp_path / "physical.npz",
        device="cpu",
        snr_db=None,
        tx_power_per_re_w=0.01,
        seed=5,
    )
    assert physical.h.shape == (1, 1, 16)
    assert physical.generation_report["power_mode"] == "absolute_tx_power_thermal_noise"
    assert physical.generation_report["noise_seed"] == 6

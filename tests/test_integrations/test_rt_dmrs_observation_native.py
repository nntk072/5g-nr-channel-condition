"""Native CUDA smoke test for RT-channel PUSCH DMRS observations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets.channel import ChannelDataset  # noqa: E402
from integrations.sionna.d1_generation_gate import authorize_d1_generation  # noqa: E402
from integrations.sionna.rt_dmrs import (  # noqa: E402
    build_rt_dmrs_generation_request,
)
from integrations.sionna.rt_dmrs import (  # noqa: E402
    build_rt_dmrs_observation_dataset as _build_rt_dmrs_observation_dataset,
)


def build_rt_dmrs_observation_dataset(datasets, **options):
    options.setdefault("destination", Path(tempfile.gettempdir()) / "loslab-native-dmrs-pilot")
    request = build_rt_dmrs_generation_request(datasets, **options)
    authorization = authorize_d1_generation(request, pilot=True)
    return _build_rt_dmrs_observation_dataset(datasets, authorization=authorization, **options)


@pytest.mark.native_sionna
def test_rt_cfr_becomes_deterministic_pusch_dmrs_receiver_observations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    frequencies = (np.arange(128) - 64) * 30_000.0
    delays_s = (20e-9, 80e-9, 160e-9, 240e-9)
    channels = np.stack([np.exp(-2j * np.pi * frequencies * delay) for delay in delays_s])
    dataset = ChannelDataset(
        h=channels[:, None, :].astype(np.complex64),
        y=np.asarray([1, 0, 1, 0], dtype=np.uint8),
        group=np.asarray([f"link-{index}" for index in range(4)]),
        domain=np.asarray(["rt_test"] * 4),
        source=np.asarray(["sionna_rt"] * 4),
        metadata=tuple(
            {
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "grid_config": {
                    "n_subcarriers": 128,
                    "spacing_hz": 30_000.0,
                },
            }
            for _ in range(4)
        ),
    )

    first = build_rt_dmrs_observation_dataset(
        [dataset], snr_db=12.0, seed=20261031, device="cuda:0"
    )
    second = build_rt_dmrs_observation_dataset(
        [dataset], snr_db=12.0, seed=20261031, device="cuda:0"
    )

    assert first.h.shape == (4, 1, 72)
    assert np.iscomplexobj(first.h) and np.isfinite(first.h).all()
    np.testing.assert_array_equal(first.h, second.h)
    np.testing.assert_array_equal(first.y, dataset.y)
    np.testing.assert_array_equal(first.group, dataset.group)
    assert first.label_semantics == "geometric_direct_path_oracle"
    assert first.generation_report["label_semantics"] == first.label_semantics
    assert first.generation_report["observation_semantics"].startswith("Sionna PUSCH DMRS")
    assert first.generation_report["receiver_metadata"]["device"] == "cuda:0"
    assert (
        first.generation_report["normalization"] == "unit mean channel power per link before AWGN"
    )


@pytest.mark.native_sionna
def test_absolute_power_dmrs_mode_preserves_link_gain_with_fixed_thermal_noise():
    channels = np.stack([np.ones(128), np.full(128, 0.5)]).astype(np.complex64)
    dataset = ChannelDataset(
        h=channels[:, None, :],
        y=np.asarray([1, 0], dtype=np.uint8),
        group=np.asarray(["unit-gain", "half-gain"]),
        domain=np.asarray(["analytic_rt_test"] * 2),
        source=np.asarray(["sionna_rt"] * 2),
        metadata=tuple(
            {
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "grid_config": {"n_subcarriers": 128, "spacing_hz": 30_000.0},
            }
            for _ in range(2)
        ),
    )

    physical = build_rt_dmrs_observation_dataset(
        [dataset],
        snr_db=None,
        tx_power_per_re_w=1e-6,
        noise_figure_db=7.0,
        temperature_k=290.0,
        seed=20261031,
        device="cpu",
    )
    normalized_unit = build_rt_dmrs_observation_dataset(
        [dataset.select(np.asarray([0]))], snr_db=12.0, seed=20261031, device="cpu"
    )
    normalized_half = build_rt_dmrs_observation_dataset(
        [dataset.select(np.asarray([1]))], snr_db=12.0, seed=20261031, device="cpu"
    )

    np.testing.assert_allclose(physical.h[1], physical.h[0] * 0.5, rtol=2e-3, atol=2e-5)
    np.testing.assert_array_equal(normalized_half.h, normalized_unit.h)
    report = physical.generation_report
    assert report["power_mode"] == "absolute_tx_power_thermal_noise"
    assert report["normalization"] == "none; LS output divided by sqrt(tx_power_per_re_w)"
    expected_noise_w = 1.380649e-23 * 290.0 * 30_000.0 * 10.0**0.7
    assert report["noise_variance_w"] == pytest.approx(expected_noise_w, rel=1e-15)
    assert report["noise_figure_db"] == 7.0
    assert report["tx_power_per_re_w"] == 1e-6

"""Native CUDA smoke test for RT-channel PUSCH DMRS observations."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets.channel import ChannelDataset  # noqa: E402
from integrations.sionna.rt_dmrs import build_rt_dmrs_observation_dataset  # noqa: E402


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
    assert first.generation_report["observation_semantics"].startswith("Sionna PUSCH DMRS")
    assert first.generation_report["receiver_metadata"]["device"] == "cuda:0"
    assert (
        first.generation_report["normalization"] == "unit mean channel power per link before AWGN"
    )

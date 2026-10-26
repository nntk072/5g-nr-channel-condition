"""Native reproducibility tests for standardized TDL/CDL generation."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.adapter import GridConfig, generate_standardized


@pytest.mark.native_sionna
def test_cdl_generation_is_reproducible_and_restores_rng_state():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from sionna.phy import config

    generator = config.torch_rng("cpu")
    generator_state = generator.get_state().clone()
    torch_state = torch.get_rng_state().clone()
    kwargs = {
        "grid": GridConfig(n_subcarriers=16, snr_db=30.0),
        "profiles": ("A",),
        "families": ("cdl",),
        "per_profile": 8,
        "seed": 8123,
        "delay_spreads_ns": (100.0,),
    }

    first = generate_standardized(**kwargs)
    assert torch.equal(generator_state, generator.get_state())
    assert torch.equal(torch_state, torch.get_rng_state())
    second = generate_standardized(**kwargs)

    np.testing.assert_array_equal(first.h, second.h)
    np.testing.assert_array_equal(first.y, second.y)
    assert all(
        row["split_subject_id"] == str(group)
        for row, group in zip(first.metadata, first.group, strict=True)
    )
    assert all(row["generation_seed"] == 8123 for row in first.metadata)
    assert len({row["channel_seed"] for row in first.metadata}) == 1


@pytest.mark.native_sionna
def test_cdl_profiles_a_through_e_generate_profile_proxy_labels():
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")

    data = generate_standardized(
        grid=GridConfig(n_subcarriers=16),
        profiles=("A", "B", "C", "D", "E"),
        families=("cdl",),
        per_profile=4,
        seed=9021,
        delay_spreads_ns=(100.0,),
    )

    assert data.h.shape == (20, 1, 16)
    assert data.label_semantics == "profile_proxy"
    assert all(np.isfinite(data.h).flat)
    for index, row in enumerate(data.metadata):
        expected_label = int(row["profile"] in "DE")
        assert row["label_semantics"] == "PROFILE_PROXY"
        assert row["channel_seed"] >= 0
        assert data.y[index] == expected_label


@pytest.mark.native_sionna
def test_standardized_generator_rejects_invalid_seed_and_spread():
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")

    with pytest.raises(ValueError, match="nonnegative integer"):
        generate_standardized(seed=-1)
    with pytest.raises(ValueError, match="finite positive"):
        generate_standardized(delay_spreads_ns=(0.0,))

"""Small integration contracts: SSB grid types, PBCH MIB fields, receiver noise and runtime probes."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.noise import proper_complex_awgn
from integrations.sionna.pbch_mib import _enum_index, _uint_bits
from integrations.sionna.ssb_grid import map_nr_ssb_resource_grid, scale_nr_ssb_resource_grid_epre
from models.real_cnn import _real_imaginary_channels


def test_ssb_grid_identities_and_scaling_are_typed() -> None:
    grid = np.ones((4, 240), dtype=np.complex128)
    with pytest.raises(TypeError, match="pss_to_sss_epre_db must be a real number"):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db="3")
    with pytest.raises(ValueError, match="ssb_grid must have shape"):
        scale_nr_ssb_resource_grid_epre(np.ones((3, 240)), pss_to_sss_epre_db=0.0)
    with pytest.raises(TypeError, match="real or complex numeric array"):
        scale_nr_ssb_resource_grid_epre(np.array([["a"] * 240] * 4), pss_to_sss_epre_db=0.0)
    bad = grid.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        scale_nr_ssb_resource_grid_epre(bad, pss_to_sss_epre_db=0.0)


def test_ssb_mapping_requires_integer_identities() -> None:
    kwargs = {
        "ssb_index_in_half_frame": 0,
        "half_frame_index": 0,
        "max_candidate_blocks_per_half_frame": 4,
    }
    with pytest.raises(TypeError, match="nid1 must be an integer"):
        map_nr_ssb_resource_grid("1", 2, np.zeros(1), **kwargs)
    with pytest.raises(TypeError, match="nid2 must be an integer"):
        map_nr_ssb_resource_grid(1, "2", np.zeros(1), **kwargs)


def test_pbch_bit_helpers_reject_wrong_types() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        _uint_bits("3", 4, "field")
    with pytest.raises(TypeError, match="must be one of"):
        _enum_index(0, ("zero", "one"), "field")
    with pytest.raises(ValueError, match="must be one of"):
        _enum_index("two", ("zero", "one"), "field")
    with pytest.raises(ValueError, match="must be in \\[0, 15\\]"):
        _uint_bits(16, 4, "field")


def test_proper_noise_requires_a_complex_reference_and_real_variances() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(1)
    with pytest.raises(TypeError, match="complex PyTorch tensor"):
        proper_complex_awgn(np.ones(3), 1.0, generator=generator)
    reference = torch.ones(3, dtype=torch.complex64)
    with pytest.raises(ValueError, match="real numeric values"):
        proper_complex_awgn(reference, "x", generator=generator)
    with pytest.raises(ValueError, match="broadcastable"):
        proper_complex_awgn(reference, torch.ones(2), generator=generator)


def test_cnn_feature_domain_rejects_real_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="complex \\[samples, links, subcarriers\\]"):
        _real_imaginary_channels(np.ones((2, 1, 4)))
    bad = np.ones((2, 1, 4), dtype=np.complex64)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="CNN inputs must be finite"):
        _real_imaginary_channels(bad)

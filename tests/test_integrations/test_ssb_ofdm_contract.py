"""Software contracts for SS/PBCH OFDM timing and frame assembly."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from integrations.sionna import ssb_ofdm as ssb_ofdm_module
from integrations.sionna.ssb_grid import map_nr_ssb_resource_grid
from integrations.sionna.ssb_ofdm import (
    nr_ssb_ofdm_demodulate,
    nr_ssb_ofdm_modulate,
    nr_ssb_ofdm_modulate_half_frame,
    nr_ssb_ofdm_modulate_radio_frame,
    nr_ssb_waveform_config,
)


def _block(candidate: int = 0, *, half_frame: int = 0):
    return map_nr_ssb_resource_grid(
        17,
        2,
        np.ones(432, dtype=np.complex128),
        ssb_index_in_half_frame=candidate,
        half_frame_index=half_frame,
        max_candidate_blocks_per_half_frame=8,
    )


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"subcarrier_spacing_hz": True}, TypeError, "real scalar"),
        ({"subcarrier_spacing_hz": float("nan")}, ValueError, "must be finite"),
        ({"subcarrier_spacing_hz": 20_000}, ValueError, "powers of two"),
        ({"candidate_start_symbol": True}, TypeError, "must be an integer"),
        ({"candidate_start_symbol": -1}, ValueError, "must fit a four-symbol"),
        ({"fft_size": True}, TypeError, "fft_size must be an integer"),
        ({"fft_size": 242}, ValueError, "cannot represent integer normal-CP"),
    ],
)
def test_ssb_waveform_config_rejects_invalid_timing_inputs(settings, error, message) -> None:
    arguments = {
        "subcarrier_spacing_hz": 15_000,
        "candidate_start_symbol": 0,
        "fft_size": 256,
    }
    arguments.update(settings)
    with pytest.raises(error, match=message):
        nr_ssb_waveform_config(**arguments)


def test_ssb_ofdm_modulator_and_demodulator_reject_bad_grids_and_waveforms() -> None:
    with pytest.raises(ValueError, match="four lengths"):
        nr_ssb_ofdm_modulate(np.zeros((4, 240)), fft_size=256, cyclic_prefix_lengths=(8,))
    with pytest.raises(TypeError, match="prefix lengths must be integers"):
        nr_ssb_ofdm_modulate(np.zeros((4, 240)), fft_size=256, cyclic_prefix_lengths=(8, 8.0, 8, 8))
    with pytest.raises(ValueError, match=r"in \[1, fft_size\)"):
        nr_ssb_ofdm_modulate(np.zeros((4, 240)), fft_size=256, cyclic_prefix_lengths=(8, 0, 8, 8))
    with pytest.raises(ValueError, match=r"shape \(4, 240\)"):
        nr_ssb_ofdm_modulate(np.zeros((4, 239)), fft_size=256, cyclic_prefix_lengths=(8,) * 4)
    with pytest.raises(TypeError, match="real or complex numeric"):
        nr_ssb_ofdm_modulate(
            np.ones((4, 240), dtype=bool), fft_size=256, cyclic_prefix_lengths=(8,) * 4
        )
    with pytest.raises(ValueError, match="must be finite"):
        nr_ssb_ofdm_modulate(
            np.full((4, 240), np.nan), fft_size=256, cyclic_prefix_lengths=(8,) * 4
        )
    with pytest.raises(ValueError, match="complex vector of shape"):
        nr_ssb_ofdm_demodulate(np.ones(1056), fft_size=256, cyclic_prefix_lengths=(8,) * 4)


@pytest.mark.parametrize(
    ("blocks", "error", "message"),
    [
        ((), ValueError, "non-empty candidate-index mapping"),
        ({True: _block()}, TypeError, "candidate indices must be integers"),
        ({8: _block()}, ValueError, "candidate indices must be in"),
        ({0: object()}, TypeError, "values must be NRSSBResourceGrid"),
        ({0: replace(_block(), pbch_dmrs=object())}, TypeError, "resource maps"),
        (
            {0: replace(_block(), physical_cell_id=0)},
            ValueError,
            "identity fields are inconsistent",
        ),
        ({0: replace(_block(), grid=np.zeros((2, 2)))}, ValueError, r"shape \(4, 240\)"),
        ({0: replace(_block(), grid=np.ones((4, 240), dtype=bool))}, TypeError, "numeric arrays"),
        ({0: replace(_block(), grid=np.full((4, 240), np.nan))}, ValueError, "must be finite"),
    ],
)
def test_half_frame_assembler_rejects_invalid_candidate_blocks(blocks, error, message) -> None:
    with pytest.raises(error, match=message):
        nr_ssb_ofdm_modulate_half_frame(
            blocks,
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        ((), TypeError, "must be a mapping"),
        ({True: {}}, TypeError, "half-frame indices must be integers"),
        ({2: {}}, ValueError, "half-frame indices must be 0 or 1"),
        ({0: ()}, TypeError, "each half-frame value must be a candidate-index mapping"),
    ],
)
def test_radio_frame_assembler_rejects_invalid_half_frame_mapping(mapping, error, message) -> None:
    with pytest.raises(error, match=message):
        nr_ssb_ofdm_modulate_radio_frame(
            mapping,
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_waveform_config_converts_extreme_subcarrier_spacing_error() -> None:
    with pytest.raises(ValueError, match="subcarrier_spacing_hz must be finite"):
        nr_ssb_waveform_config(
            subcarrier_spacing_hz=10**1000,
            candidate_start_symbol=0,
            fft_size=256,
        )


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        (
            {0: replace(_block(), pbch_dmrs=replace(_block().pbch_dmrs, half_frame_index=2))},
            "half-frame index must be 0 or 1",
        ),
        (
            {0: _block(), 1: replace(_block(1), pbch_dmrs=object())},
            "resource maps",
        ),
        (
            {
                0: _block(),
                1: map_nr_ssb_resource_grid(
                    18,
                    0,
                    np.ones(432),
                    ssb_index_in_half_frame=1,
                    half_frame_index=0,
                    max_candidate_blocks_per_half_frame=8,
                ),
            },
            "same physical cell ID",
        ),
        (
            {0: _block(), 1: _block(1, half_frame=1)},
            "same half-frame index",
        ),
    ],
)
def test_half_frame_assembler_checks_shared_candidate_metadata(blocks, message) -> None:
    with pytest.raises(ValueError if message != "resource maps" else TypeError, match=message):
        nr_ssb_ofdm_modulate_half_frame(
            blocks,
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_half_frame_assembler_rejects_block_waveform_length_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        ssb_ofdm_module, "nr_ssb_ofdm_modulate", lambda *_args, **_kwargs: np.zeros(1)
    )
    with pytest.raises(RuntimeError, match="do not align with the half-frame symbol clock"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: _block()},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_radio_frame_assembler_rejects_unknown_inner_grid() -> None:
    with pytest.raises(TypeError, match="SSB grids must be NRSSBResourceGrid"):
        nr_ssb_ofdm_modulate_radio_frame(
            {0: {0: object()}},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_ofdm_demodulator_rejects_nonfinite_waveform() -> None:
    waveform = np.ones(4 * (256 + 8), dtype=np.complex128)
    waveform[0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError, match="waveform must be finite"):
        nr_ssb_ofdm_demodulate(waveform, fft_size=256, cyclic_prefix_lengths=(8,) * 4)
    with pytest.raises(ValueError, match="fft_size must be even and greater than 240"):
        nr_ssb_ofdm_modulate(np.zeros((4, 240)), fft_size=241, cyclic_prefix_lengths=(8,) * 4)


def test_half_frame_checks_cp_representability_and_consistent_block_length(monkeypatch) -> None:
    with pytest.raises(ValueError, match="cannot represent integer normal-CP"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: _block()},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=242,
        )
    monkeypatch.setattr(
        ssb_ofdm_module, "nr_ssb_ofdm_modulate", lambda *_args, **_kwargs: np.zeros(1)
    )
    with pytest.raises(RuntimeError, match="do not align with the half-frame symbol clock"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: _block()},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_half_frame_timing_sum_is_checked(monkeypatch) -> None:
    original_cumsum = np.cumsum

    def shifted_cumsum(values, *, out=None):
        result = original_cumsum(values)
        result[-1] += 1
        out[:] = result
        return out

    monkeypatch.setattr(ssb_ofdm_module.np, "cumsum", shifted_cumsum)
    with pytest.raises(RuntimeError, match="do not sum to a 5 ms half-frame"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: _block()},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_half_frame_rejects_invalid_spacing_type_after_timing_selection(monkeypatch) -> None:
    monkeypatch.setattr(ssb_ofdm_module, "ssb_candidate_symbols", lambda *_args, **_kwargs: (0,))
    with pytest.raises(TypeError, match="subcarrier_spacing_hz must be a real scalar"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: _block()},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=True,
            frequency_range="FR1",
            fft_size=256,
        )


def test_radio_frame_validates_subcarrier_spacing_after_timing_selection(monkeypatch) -> None:
    monkeypatch.setattr(ssb_ofdm_module, "ssb_candidate_symbols", lambda *_args, **_kwargs: (0,))
    base = {
        "ssb_blocks_by_half_frame": {},
        "case": "A",
        "carrier_frequency_hz": 3.5e9,
        "frequency_range": "FR1",
        "fft_size": 256,
    }
    for spacing, error, message in (
        (True, TypeError, "real scalar"),
        (15_000.5, ValueError, "finite integer"),
        (20_000, ValueError, "unsupported SSB subcarrier spacing"),
    ):
        with pytest.raises(error, match=message):
            nr_ssb_ofdm_modulate_radio_frame(**base, subcarrier_spacing_hz=spacing)


def test_radio_frame_checks_half_frame_builder_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        ssb_ofdm_module,
        "nr_ssb_ofdm_modulate_half_frame",
        lambda *_args, **_kwargs: SimpleNamespace(
            half_frame_index=0,
            subcarrier_spacing_hz=30_000,
            fft_size=256,
            sample_rate_hz=256 * 15_000,
            candidate_start_symbols=(0,),
            waveform=np.zeros(256 * 15_000 // 200),
            transmitted_candidate_indices=(),
        ),
    )
    with pytest.raises(ValueError, match="configuration is inconsistent"):
        nr_ssb_ofdm_modulate_radio_frame(
            {0: {0: _block()}},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )

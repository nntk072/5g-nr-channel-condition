"""Shared DM-RS reference and interpolation types for the native PUSCH tests."""

import math
from typing import Any, TypedDict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")


pytest.importorskip("sionna")


class _InterpolationConfig(TypedDict, total=False):
    n_size_bwp: int
    subcarrier_spacing_khz: int
    dmrs_config_type: int
    dmrs_length: int
    dmrs_additional_position: int
    num_cdm_groups_without_data: int


def _pusch_dmrs_gold_reference(
    *,
    cell_id: int,
    dmrs_n_id: tuple[int, int] | None,
    n_scid: int,
    slot: int,
    symbol: int,
    count: int,
) -> Any:
    """Independent TS 38.211 5.2.1/6.4.1.1.1 Gold-sequence reference.

    For transform-precoding-disabled PUSCH, use
    ``c_init = 2^17(14*n_slot + l + 1)(2*N_ID^nSCID + 1)
    + 2*N_ID^nSCID + n_SCID (mod 2^31)`` and map consecutive Gold chips to
    ``(1-2*c(2n) + j*(1-2*c(2n+1)))/sqrt(2)``.
    """
    n_id = cell_id if dmrs_n_id is None else dmrs_n_id[n_scid]
    c_init = (2**17 * (14 * slot + symbol + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % 2**31
    n_c = 1600
    sequence_length = 2 * count
    state_length = n_c + sequence_length + 31
    x1 = [0] * state_length
    x2 = [0] * state_length
    x1[0] = 1
    for index in range(31):
        x2[index] = (c_init >> index) & 1
    for index in range(n_c + sequence_length):
        x1[index + 31] = (x1[index + 3] + x1[index]) & 1
        x2[index + 31] = (x2[index + 3] + x2[index + 2] + x2[index + 1] + x2[index]) & 1
    chips = [(x1[index + n_c] + x2[index + n_c]) & 1 for index in range(sequence_length)]
    scale = 1.0 / math.sqrt(2.0)
    return torch.tensor(
        [
            complex(scale * (1 - 2 * chips[2 * index]), scale * (1 - 2 * chips[2 * index + 1]))
            for index in range(count)
        ],
        dtype=torch.complex128,
    )

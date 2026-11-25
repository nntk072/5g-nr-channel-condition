"""SS/PBCH block scaling and BCCH-BCH message dispatch: shape, dtype, finiteness and alternative checks."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import pbch_dmrs
from integrations.sionna.pbch_mib import (
    PBCHMessageClassExtension,
    nr_bcch_bch_message_parse,
    nr_bcch_bch_message_serialize,
    nr_pbch_mib_serialize,
)
from integrations.sionna.ssb_grid import scale_nr_ssb_resource_grid_epre


def test_pbch_dmrs_coordinate_count_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pbch_dmrs, "_DMRS_SYMBOL_COUNT", 145)

    with pytest.raises(RuntimeError, match="coordinate count is inconsistent"):
        pbch_dmrs._pbch_dmrs_coordinates(0)


def test_epre_ratio_must_be_zero_or_three_db() -> None:
    grid = np.ones((4, 240), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 0 or 3 dB"):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=1.5)
    with pytest.raises(TypeError, match="must be a real number"):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=True)


@pytest.mark.parametrize(
    ("grid", "error", "message"),
    [
        pytest.param(np.ones((4, 239)), ValueError, "must have shape", id="shape"),
        pytest.param(np.zeros((4, 240), dtype=bool), TypeError, "real or complex", id="boolean"),
        pytest.param(
            np.where(np.arange(4 * 240).reshape(4, 240) == 3, np.nan, 1.0),
            ValueError,
            "must be finite",
            id="nan",
        ),
    ],
)
def test_ssb_grid_shape_dtype_and_finiteness(grid, error, message) -> None:
    with pytest.raises(error, match=message):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=0.0)


def test_bcch_bch_alternatives_round_trip_through_their_serializer() -> None:
    extension = PBCHMessageClassExtension()
    bits = nr_bcch_bch_message_serialize(extension)
    assert bits.shape == (24,)
    assert bits[0] == 1
    restored = nr_bcch_bch_message_parse(bits)
    assert isinstance(restored, PBCHMessageClassExtension)


def test_unknown_message_alternatives_are_refused() -> None:
    with pytest.raises(TypeError, match="must be PBCHMIB or PBCHMessageClassExtension"):
        nr_bcch_bch_message_serialize("not-a-message")
    with pytest.raises(TypeError, match="mib must be a PBCHMIB"):
        nr_pbch_mib_serialize("not-a-mib")

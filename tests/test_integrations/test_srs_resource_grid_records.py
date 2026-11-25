"""SRS resource-grid record invariants, independent of mapping algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.srs.srs_constants.resource_grids import (
    NRSRSHoppedMultiportResourceGrid,
    NRSRSHoppedResourceGrid,
    NRSRSMultiportResourceGrid,
    NRSRSPositioningMultiportResourceGrid,
    NRSRSPositioningResourceGrid,
    NRSRSResourceGrid,
)


def _single_grid_args():
    grid = np.ones((2, 3), dtype=np.complex128)
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    return {"grid": grid, "mask": mask, "symbol_indices": (0,), "subcarrier_indices": (1,)}


def _hopped_grid_args():
    grid = np.ones((2, 3), dtype=np.complex128)
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    mask[1, 2] = True
    return {
        "grid": grid,
        "mask": mask,
        "symbol_indices": (0, 1),
        "subcarrier_indices_by_symbol": ((1,), (2,)),
        "position_indices_by_symbol": ((0,), (0,)),
        "n_srs_by_symbol": (None, 0),
        "positioning_offsets": (0, 0),
    }


def _hopped_multiport_args():
    grid = np.ones((2, 2, 3), dtype=np.complex128)
    mask = np.zeros((2, 2, 3), dtype=bool)
    mask[:, 0, 1] = True
    mask[:, 1, 2] = True
    return {**_hopped_grid_args(), "grid": grid, "mask": mask}


def _positioning_args():
    args = _hopped_grid_args()
    return {
        "grid": args["grid"],
        "mask": args["mask"],
        "symbol_indices": args["symbol_indices"],
        "subcarrier_indices_by_symbol": args["subcarrier_indices_by_symbol"],
        "positioning_offsets": args["positioning_offsets"],
    }


def _positioning_multiport_args():
    args = _positioning_args()
    grid = np.ones((2, 2, 3), dtype=np.complex128)
    mask = np.zeros((2, 2, 3), dtype=bool)
    mask[:, 0, 1] = True
    mask[:, 1, 2] = True
    return {**args, "grid": grid, "mask": mask}


def _multiport_args():
    args = _single_grid_args()
    grid = np.ones((2, 2, 3), dtype=np.complex128)
    mask = np.zeros((2, 2, 3), dtype=bool)
    mask[:, 0, 1] = True
    return {**args, "grid": grid, "mask": mask}


@pytest.mark.parametrize(
    ("record_type", "factory", "changes", "error", "message"),
    [
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"grid": np.ones((2, 3))},
            ValueError,
            "grid must be complex",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"mask": np.ones((2, 3), dtype=np.uint8)},
            ValueError,
            "mask must be boolean",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"grid": np.full((2, 3), complex(np.inf, 0))},
            ValueError,
            "grid must be finite",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"symbol_indices": ()},
            ValueError,
            "indices must be nonempty",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"subcarrier_indices": (True,)},
            TypeError,
            "indices must be integers",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"symbol_indices": (0, 0)},
            ValueError,
            "unique and inside",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"subcarrier_indices": (3,)},
            ValueError,
            "unique and inside",
        ),
        (
            NRSRSResourceGrid,
            _single_grid_args,
            {"mask": np.zeros((2, 3), dtype=bool)},
            ValueError,
            "mask disagrees",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"grid": np.ones((2, 3))},
            ValueError,
            "grid must be complex",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"mask": np.ones((2, 3), dtype=np.uint8)},
            ValueError,
            "mask must be boolean",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"grid": np.full((2, 3), complex(np.inf, 0))},
            ValueError,
            "grid must be finite",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"n_srs_by_symbol": (None,)},
            ValueError,
            "rosters must align",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"position_indices_by_symbol": ((), ())},
            ValueError,
            "rosters must be nonempty",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"positioning_offsets": (False, 0)},
            TypeError,
            "indices and counters must be integers",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"n_srs_by_symbol": ("bad", 0)},
            TypeError,
            "counters must be integers or None",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"symbol_indices": (0, 0)},
            ValueError,
            "nonnegative and in-grid",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"subcarrier_indices_by_symbol": ((1, 1), (2,))},
            ValueError,
            "nonnegative and in-grid",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"positioning_offsets": (-1, 0)},
            ValueError,
            "nonnegative and in-grid",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"n_srs_by_symbol": (-1, 0)},
            ValueError,
            "nonnegative and in-grid",
        ),
        (
            NRSRSHoppedResourceGrid,
            _hopped_grid_args,
            {"mask": np.zeros((2, 3), dtype=bool)},
            ValueError,
            "mask disagrees",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"grid": np.ones((2, 2, 3))},
            ValueError,
            "grid must be complex",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"grid": np.empty((0, 2, 3), dtype=np.complex128)},
            ValueError,
            "must have ports",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"mask": np.ones((2, 2, 3), dtype=np.uint8)},
            ValueError,
            "mask must be boolean",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"n_srs_by_symbol": (None,)},
            ValueError,
            "rosters must align",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"position_indices_by_symbol": ((), ())},
            ValueError,
            "rosters must be nonempty",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"symbol_indices": (False, 1)},
            TypeError,
            "indices must be integers",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"n_srs_by_symbol": ("bad", 0)},
            TypeError,
            "counters must be integers or None",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"positioning_offsets": (-1, 0)},
            ValueError,
            "nonnegative and in-grid",
        ),
        (
            NRSRSHoppedMultiportResourceGrid,
            _hopped_multiport_args,
            {"mask": np.zeros((2, 2, 3), dtype=bool)},
            ValueError,
            "mask disagrees",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"grid": np.ones((2, 3))},
            ValueError,
            "grid must be complex",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"mask": np.ones((2, 3), dtype=np.uint8)},
            ValueError,
            "mask must be boolean",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"grid": np.full((2, 3), complex(np.inf, 0))},
            ValueError,
            "grid must be finite",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"positioning_offsets": (0,)},
            ValueError,
            "rosters must align",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"subcarrier_indices_by_symbol": ((), (2,))},
            ValueError,
            "at least one subcarrier",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"symbol_indices": (False, 1)},
            TypeError,
            "indices must be integers",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"positioning_offsets": (-1, 0)},
            ValueError,
            "unique and inside",
        ),
        (
            NRSRSPositioningResourceGrid,
            _positioning_args,
            {"mask": np.zeros((2, 3), dtype=bool)},
            ValueError,
            "mask disagrees",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"grid": np.ones((2, 2, 3))},
            ValueError,
            "grid must be complex",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"grid": np.empty((0, 2, 3), dtype=np.complex128)},
            ValueError,
            "must have ports",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"mask": np.ones((2, 2, 3), dtype=np.uint8)},
            ValueError,
            "mask must be boolean",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"positioning_offsets": (0,)},
            ValueError,
            "rosters must align",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"subcarrier_indices_by_symbol": ((), (2,))},
            ValueError,
            "at least one subcarrier",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"symbol_indices": (False, 1)},
            TypeError,
            "indices must be integers",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"positioning_offsets": (-1, 0)},
            ValueError,
            "unique and inside",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"active_port_symbols": ((True,),)},
            ValueError,
            "active_port_symbols must be boolean",
        ),
        (
            NRSRSPositioningMultiportResourceGrid,
            _positioning_multiport_args,
            {"mask": np.zeros((2, 2, 3), dtype=bool)},
            ValueError,
            "mask disagrees",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"grid": np.ones((2, 2, 3))},
            ValueError,
            "grid must be complex",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"grid": np.empty((0, 2, 3), dtype=np.complex128)},
            ValueError,
            "must have ports",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"mask": np.ones((2, 2, 3), dtype=np.uint8)},
            ValueError,
            "mask must be boolean",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"symbol_indices": ()},
            ValueError,
            "indices must be nonempty",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"subcarrier_indices": (False,)},
            TypeError,
            "indices must be integers",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"symbol_indices": (0, 0)},
            ValueError,
            "unique and inside",
        ),
        (
            NRSRSMultiportResourceGrid,
            _multiport_args,
            {"active_port_symbols": ((True,),)},
            ValueError,
            "active_port_symbols must be boolean",
        ),
    ],
)
def test_srs_resource_grid_record_rejects_malformed_invariants(
    record_type, factory, changes, error, message
):
    arguments = factory()
    arguments.update(changes)
    with pytest.raises(error, match=message):
        record_type(**arguments)


@pytest.mark.parametrize(
    ("record_type", "factory"),
    [
        (NRSRSResourceGrid, _single_grid_args),
        (NRSRSHoppedResourceGrid, _hopped_grid_args),
        (NRSRSHoppedMultiportResourceGrid, _hopped_multiport_args),
        (NRSRSPositioningResourceGrid, _positioning_args),
        (NRSRSPositioningMultiportResourceGrid, _positioning_multiport_args),
        (NRSRSMultiportResourceGrid, _multiport_args),
    ],
)
def test_srs_resource_grid_records_freeze_views_and_normalize_indices(record_type, factory):
    record = record_type(**factory())
    assert not record.grid.flags.writeable
    assert not record.mask.flags.writeable

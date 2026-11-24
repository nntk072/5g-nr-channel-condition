"""Framework-independent contracts for NR PUSCH receiver observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from receivers.nr.pusch_core.constants import (
    MAX_CARRIER_GRID_START_CRB,
    MAX_CRB_END_EXCLUSIVE,
)
from receivers.nr.pusch_core.dmrs import (
    _pusch_dmrs_gold_sequence,
    build_pusch_intra_slot_hop_dmrs_grid,
    derive_pusch_intra_slot_hop_dmrs_plan,
    verify_pusch_intra_slot_hop_dmrs_grid,
    verify_pusch_intra_slot_hop_dmrs_plan,
)
from receivers.nr.pusch_core.estimation import (
    estimate_demap_pusch_intra_slot_hop,
)
from receivers.nr.pusch_core.hopping import (
    _array_is_finite_c_order,
    _sha256_array_c_order,
    compose_pusch_intra_slot_hops,
    dehop_pusch_intra_slot_received_grid,
    verify_pusch_intra_slot_dehopped_grid,
    verify_pusch_intra_slot_hops,
)
from receivers.nr.pusch_core.mapping import (
    _nearest_sorted_positions,
    _normalized_hop_noise_variance,
    map_pusch_intra_slot_hop_data_symbols,
    verify_pusch_intra_slot_hop_allocation_grid,
)
from receivers.nr.pusch_core.types import (
    NRPUSCHDehoppedGrid,
    NRPUSCHFrequencyHopGrid,
    NRPUSCHHoppedAllocationGrid,
    NRPUSCHHoppedEstimateDemap,
    NRPUSCHIntraSlotHopDMRSGrid,
    NRPUSCHIntraSlotHopDMRSPlan,
)


@dataclass(frozen=True, slots=True)
class NRPUSCHConfig:
    """PUSCH/DMRS resource-grid and channel-estimator configuration.

    Subcarrier spacing is in kHz and BWP size is in resource blocks.
    Sionna validates additional cross-field constraints when the optional
    adapter is constructed.
    """

    n_size_bwp: int = 4
    subcarrier_spacing_khz: int = 30
    num_antenna_ports: int = 1
    num_layers: int = 1
    n_cell_id: int = 1
    dmrs_n_id: tuple[int, int] | None = None
    dmrs_n_scid: int = 0
    dmrs_config_type: int = 1
    dmrs_length: int = 1
    dmrs_additional_position: int = 0
    num_cdm_groups_without_data: int = 1
    mapping_type: str = "A"
    symbol_allocation: tuple[int, int] = (0, 14)
    precoding: str = "non-codebook"
    interpolation_type: str = "lin"
    n_size_grid: int | None = None
    n_start_grid: int = 0
    n_start_bwp: int = 0
    frequency_hopping: str = "neither"
    transform_precoding: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "n_size_bwp",
            "subcarrier_spacing_khz",
            "n_start_grid",
            "n_start_bwp",
            "num_antenna_ports",
            "num_layers",
            "n_cell_id",
            "dmrs_n_scid",
            "dmrs_config_type",
            "dmrs_length",
            "dmrs_additional_position",
            "num_cdm_groups_without_data",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.n_size_bwp <= 275:
            raise ValueError("n_size_bwp must be in [1, 275]")
        n_size_grid = self.n_size_grid
        if n_size_grid is None:
            n_size_grid = self.n_size_bwp
            object.__setattr__(self, "n_size_grid", n_size_grid)
        elif isinstance(n_size_grid, bool) or not isinstance(n_size_grid, int):
            raise TypeError("n_size_grid must be an integer or None")
        if not 1 <= n_size_grid <= 275:
            raise ValueError("n_size_grid must be in [1, 275]")
        if not 0 <= self.n_start_grid <= 2199:
            raise ValueError("n_start_grid must be in [0, 2199]")
        if not 0 <= self.n_start_bwp <= 2473:
            raise ValueError("n_start_bwp must be in [0, 2473]")
        bwp_start_in_grid = self.n_start_bwp - self.n_start_grid
        if bwp_start_in_grid < 0 or bwp_start_in_grid + self.n_size_bwp > n_size_grid:
            raise ValueError("BWP must lie within the configured carrier resource grid")
        if self.subcarrier_spacing_khz not in {15, 30, 60, 120, 240, 480, 960}:
            raise ValueError("Unsupported NR subcarrier spacing")
        if self.num_antenna_ports not in {1, 2, 4}:
            raise ValueError("num_antenna_ports must be 1, 2, or 4")
        if not 1 <= self.num_layers <= self.num_antenna_ports:
            raise ValueError("num_layers must be between 1 and num_antenna_ports")
        if not 0 <= self.n_cell_id <= 1007:
            raise ValueError("n_cell_id must be in [0, 1007]")
        if self.dmrs_n_id is not None:
            if not isinstance(self.dmrs_n_id, tuple) or len(self.dmrs_n_id) != 2:
                raise TypeError("dmrs_n_id must be a two-element tuple or None")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535
                for value in self.dmrs_n_id
            ):
                raise ValueError("each dmrs_n_id value must be an integer in [0, 65535]")
        if self.dmrs_n_scid not in {0, 1}:
            raise ValueError("dmrs_n_scid must be 0 or 1")
        if self.dmrs_config_type not in {1, 2}:
            raise ValueError("dmrs_config_type must be 1 or 2")
        if self.dmrs_length not in {1, 2}:
            raise ValueError("dmrs_length must be 1 or 2")
        max_position = 3 if self.dmrs_length == 1 else 1
        if not 0 <= self.dmrs_additional_position <= max_position:
            raise ValueError(f"dmrs_additional_position must be in [0, {max_position}]")
        if not 1 <= self.num_cdm_groups_without_data <= 3:
            raise ValueError("num_cdm_groups_without_data must be in [1, 3]")
        if self.mapping_type not in {"A", "B"}:
            raise ValueError("mapping_type must be A or B")
        if not isinstance(self.frequency_hopping, str) or self.frequency_hopping != "neither":
            raise ValueError(
                "frequency_hopping currently supports only 'neither'; "
                "PUSCH frequency hopping is not implemented by this receiver"
            )
        if type(self.transform_precoding) is not bool:
            raise TypeError("transform_precoding must be a Boolean")
        if self.transform_precoding:
            raise ValueError(
                "transform_precoding is not supported: pinned Sionna PHY 2.2.0 "
                "does not implement DFT-s-OFDM PUSCH"
            )
        if (
            not isinstance(self.symbol_allocation, (tuple, list))
            or len(self.symbol_allocation) != 2
        ):
            raise TypeError("symbol_allocation must be a two-element tuple or list")
        allocation = tuple(self.symbol_allocation)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in allocation):
            raise TypeError("symbol_allocation values must be integers")
        object.__setattr__(self, "symbol_allocation", allocation)
        start_symbol, duration = allocation
        if not 0 <= start_symbol < 14 or not 1 <= duration <= 14 - start_symbol:
            raise ValueError("symbol_allocation must fit within the 14-symbol slot")
        if self.mapping_type == "A" and start_symbol != 0:
            raise ValueError("PUSCH mapping type A requires symbol allocation to start at 0")
        if self.precoding not in {"codebook", "non-codebook"}:
            raise ValueError("precoding must be codebook or non-codebook")
        if self.interpolation_type not in {"nn", "lin", "lin_time_avg"}:
            raise ValueError("Unsupported Sionna PUSCH interpolation type")

    @property
    def carrier_grid_size(self) -> int:
        """Carrier resource-block count, defaulting to the complete configured BWP."""
        return self.n_size_bwp if self.n_size_grid is None else self.n_size_grid


@dataclass(frozen=True, slots=True)
class NRChannelEstimate:
    """Estimated channel, broadcastable estimator variance, pilot mask, and metadata.

    Channel axes remain [batch, rx, rx antenna, tx, stream, OFDM symbol,
    effective subcarrier]; no antenna or time axis is flattened.
    """

    h_hat: Any
    error_variance: Any
    pilot_mask: Any
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        h_shape = tuple(getattr(self.h_hat, "shape", ()))
        error_shape = tuple(getattr(self.error_variance, "shape", ()))
        mask_shape = tuple(getattr(self.pilot_mask, "shape", ()))
        if len(h_shape) != 7:
            raise ValueError("h_hat must have seven named channel axes")
        if any(size <= 0 for size in h_shape):
            raise ValueError("h_hat axes must be non-empty")
        if len(error_shape) != 7 or any(
            variance_axis not in (1, estimate_axis)
            for variance_axis, estimate_axis in zip(error_shape, h_shape, strict=True)
        ):
            raise ValueError("error_variance must be broadcastable to h_hat")
        if any(size <= 0 for size in error_shape):
            raise ValueError("error_variance axes must be non-empty")
        if len(mask_shape) != 4 or mask_shape != h_shape[3:]:
            raise ValueError("pilot_mask must have [tx, stream, symbol, subcarrier] axes")
        if not isinstance(self.metadata, Mapping) or not self.metadata:
            raise ValueError("estimate metadata must not be empty")

        torch_module: Any | None = None
        if any(
            type(value).__module__.partition(".")[0] == "torch"
            for value in (self.h_hat, self.error_variance, self.pilot_mask)
        ):
            try:
                import torch as imported_torch
            except ImportError:
                pass
            else:
                torch_module = imported_torch

        def array_domain(value: Any, name: str) -> tuple[str, bool, bool]:
            if isinstance(value, np.ndarray):
                kind = value.dtype.kind
                if kind not in "biufc":
                    raise TypeError(f"{name} must be a dense numeric array")
                finite = bool(np.isfinite(value).all())
                negative = bool(np.any(value < 0)) if kind in "iuf" else False
                return kind, finite, negative
            if torch_module is not None and torch_module.is_tensor(value):
                if value.layout != torch_module.strided:
                    raise TypeError(f"{name} must be a dense numeric tensor")
                is_complex = value.is_complex()
                is_bool = value.dtype == torch_module.bool
                is_real = value.is_floating_point() or (not is_complex and not is_bool)
                if not (is_complex or is_bool or is_real):
                    raise TypeError(f"{name} must be a dense numeric tensor")
                finite = bool(torch_module.isfinite(value).all())
                negative = bool((value < 0).any()) if is_real and not is_bool else False
                if is_complex:
                    kind = "c"
                elif is_bool:
                    kind = "b"
                elif value.is_floating_point():
                    kind = "f"
                else:
                    kind = "i"
                return kind, finite, negative
            raise TypeError(f"{name} must be a NumPy array or dense PyTorch tensor")

        h_kind, h_finite, _ = array_domain(self.h_hat, "h_hat")
        if h_kind != "c":
            raise TypeError("h_hat must be complex-valued")
        if not h_finite:
            raise ValueError("h_hat must contain only finite values")
        variance_kind, variance_finite, variance_negative = array_domain(
            self.error_variance, "error_variance"
        )
        if variance_kind not in "uf":
            raise TypeError("error_variance must be real-valued")
        if not variance_finite:
            raise ValueError("error_variance must contain only finite values")
        if variance_negative:
            raise ValueError("error_variance must be nonnegative")
        mask_kind, mask_finite, _ = array_domain(self.pilot_mask, "pilot_mask")
        if mask_kind != "b":
            raise TypeError("pilot_mask must be Boolean-valued")
        if not mask_finite:
            raise ValueError("pilot_mask must contain only finite values")


__all__ = [
    "MAX_CARRIER_GRID_START_CRB",
    "MAX_CRB_END_EXCLUSIVE",
    "NRChannelEstimate",
    "NRPUSCHConfig",
    "NRPUSCHDehoppedGrid",
    "NRPUSCHFrequencyHopGrid",
    "NRPUSCHHoppedAllocationGrid",
    "NRPUSCHHoppedEstimateDemap",
    "NRPUSCHIntraSlotHopDMRSGrid",
    "NRPUSCHIntraSlotHopDMRSPlan",
    "_array_is_finite_c_order",
    "_nearest_sorted_positions",
    "_normalized_hop_noise_variance",
    "_pusch_dmrs_gold_sequence",
    "_sha256_array_c_order",
    "build_pusch_intra_slot_hop_dmrs_grid",
    "compose_pusch_intra_slot_hops",
    "dehop_pusch_intra_slot_received_grid",
    "derive_pusch_intra_slot_hop_dmrs_plan",
    "estimate_demap_pusch_intra_slot_hop",
    "map_pusch_intra_slot_hop_data_symbols",
    "verify_pusch_intra_slot_dehopped_grid",
    "verify_pusch_intra_slot_hop_allocation_grid",
    "verify_pusch_intra_slot_hop_dmrs_grid",
    "verify_pusch_intra_slot_hop_dmrs_plan",
    "verify_pusch_intra_slot_hops",
]

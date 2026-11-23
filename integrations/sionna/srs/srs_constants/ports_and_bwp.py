"""SRS port configuration and bandwidth-part selection records."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, slots=True)
class NRSRSPortConfiguration:
    antenna_ports: tuple[int, ...]
    ports8_tdm: bool

    def __post_init__(self) -> None:
        if not isinstance(self.antenna_ports, tuple) or not self.antenna_ports:
            raise TypeError("antenna_ports must be a nonempty tuple")
        if any(
            isinstance(port, bool) or not isinstance(port, Integral) for port in self.antenna_ports
        ):
            raise TypeError("antenna_ports must contain integers")
        if not isinstance(self.ports8_tdm, bool):
            raise TypeError("ports8_tdm must be a boolean")
        ports = tuple(int(port) for port in self.antenna_ports)
        if ports not in {
            (1000,),
            (1000, 1001),
            (1000, 1001, 1002, 1003),
            tuple(range(1000, 1008)),
        }:
            raise ValueError("antenna_ports must be a complete ordered SRS port roster")
        if self.ports8_tdm and ports != tuple(range(1000, 1008)):
            raise ValueError("ports8_tdm requires the complete eight-port roster")
        object.__setattr__(self, "antenna_ports", ports)


@dataclass(frozen=True, slots=True)
class NRSRSBWPSelection:
    """Per-cell active UL BWP result for a decoded DCI 0_3 indicator."""

    indicator_bit_width: int
    selected_bwp_id_by_cell: tuple[tuple[str, int], ...]
    outcome_by_cell: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.indicator_bit_width, bool)
            or not isinstance(self.indicator_bit_width, Integral)
            or not 0 <= self.indicator_bit_width <= 2
        ):
            raise ValueError("indicator_bit_width must be in [0, 2]")
        selected = tuple(self.selected_bwp_id_by_cell)
        outcomes = tuple(self.outcome_by_cell)
        if not selected or len(selected) != len(outcomes):
            raise ValueError("BWP selection and outcome rows must be nonempty and aligned")
        if tuple(cell for cell, _ in selected) != tuple(cell for cell, _ in outcomes):
            raise ValueError("BWP selection and outcome rows must have identical cell order")
        if any(
            not isinstance(cell, str)
            or not cell
            or isinstance(bwp_id, bool)
            or not isinstance(bwp_id, Integral)
            or not 0 <= bwp_id <= 4
            for cell, bwp_id in selected
        ):
            raise ValueError("BWP selection rows require cell IDs and BWP IDs in [0, 4]")
        valid_outcomes = {
            "field_absent",
            "single_bwp",
            "selected_by_indicator",
            "codepoint_not_configured_for_cell",
            "active_bwp_change_unsupported",
        }
        if any(
            not isinstance(cell, str)
            or not cell
            or not isinstance(outcome, str)
            or outcome not in valid_outcomes
            for cell, outcome in outcomes
        ):
            raise ValueError("BWP outcome rows contain an invalid status")
        object.__setattr__(self, "indicator_bit_width", int(self.indicator_bit_width))
        object.__setattr__(
            self, "selected_bwp_id_by_cell", tuple((cell, int(bwp_id)) for cell, bwp_id in selected)
        )
        object.__setattr__(self, "outcome_by_cell", outcomes)


@dataclass(frozen=True, slots=True)
class NRSRSBWPFrequencyBounds:
    """UL BWP extent in the caller's common subcarrier-index coordinate system."""

    start_subcarrier: int
    size_subcarriers: int

    def __post_init__(self) -> None:
        for name in ("start_subcarrier", "size_subcarriers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.start_subcarrier < 0 or self.size_subcarriers <= 0:
            raise ValueError("BWP start must be nonnegative and size must be positive")
        object.__setattr__(self, "start_subcarrier", int(self.start_subcarrier))
        object.__setattr__(self, "size_subcarriers", int(self.size_subcarriers))

    @property
    def end_subcarrier_exclusive(self) -> int:
        return self.start_subcarrier + self.size_subcarriers

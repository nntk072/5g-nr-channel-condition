"""Optional Sionna PHY adapter for NR PUSCH DMRS channel estimation."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from datasets.nr import NRObservation
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from .random import call_with_sionna_seed


class SionnaPUSCHLSReceiver:
    """Estimate channels from a received PUSCH resource grid using NR DMRS.

    Sionna and PyTorch are imported only at construction time.
    """

    def __init__(self, config: NRPUSCHConfig, *, device: str = "cpu") -> None:
        try:
            import torch
            from sionna.phy import config as phy_config
            from sionna.phy.nr import PUSCHConfig, PUSCHLSChannelEstimator, PUSCHTransmitter
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional Sionna 2.2 PHY dependencies to use this receiver"
            ) from exc

        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be cpu or a CUDA device such as cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        phy_config.device = device
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = config.subcarrier_spacing_khz
        pusch_config.carrier.n_size_grid = config.n_size_bwp
        pusch_config.carrier.n_cell_id = config.n_cell_id
        pusch_config.n_size_bwp = config.n_size_bwp
        pusch_config.num_antenna_ports = config.num_antenna_ports
        pusch_config.num_layers = config.num_layers
        pusch_config.dmrs.config_type = config.dmrs_config_type
        pusch_config.dmrs.length = config.dmrs_length
        pusch_config.dmrs.additional_position = config.dmrs_additional_position
        pusch_config.dmrs.num_cdm_groups_without_data = config.num_cdm_groups_without_data
        pusch_config.dmrs.n_id = config.dmrs_n_id
        pusch_config.dmrs.n_scid = config.dmrs_n_scid
        pusch_config.mapping_type = config.mapping_type
        pusch_config.symbol_allocation = list(config.symbol_allocation)
        pusch_config.precoding = config.precoding

        self.config = config
        self.device = device
        self._torch = torch
        self._transmitter = PUSCHTransmitter(
            pusch_config, return_bits=True, output_domain="freq", device=device
        )
        self._resource_grid = self._transmitter.resource_grid
        self._pilot_mask = self._resource_grid.pilot_pattern.mask
        self._estimator = PUSCHLSChannelEstimator(
            self._resource_grid,
            dmrs_length=config.dmrs_length,
            dmrs_additional_position=config.dmrs_additional_position,
            num_cdm_groups_without_data=config.num_cdm_groups_without_data,
            interpolation_type=config.interpolation_type,
            device=device,
        )
        try:
            sionna_version = version("sionna")
        except PackageNotFoundError:
            sionna_version = "unknown"
        self._metadata: dict[str, Any] = {
            **asdict(config),
            "sionna_version": sionna_version,
            "receiver": "PUSCHLSChannelEstimator",
            "device": device,
            "resource_grid": {
                "num_ofdm_symbols": self._resource_grid.num_ofdm_symbols,
                "fft_size": self._resource_grid.fft_size,
                "subcarrier_spacing_hz": config.subcarrier_spacing_khz * 1000,
            },
        }

    @property
    def resource_grid(self) -> Any:
        """Sionna resource grid defining the expected observation shape."""
        return self._resource_grid

    @property
    def pilot_mask(self) -> Any:
        """Boolean [tx, stream, symbol, subcarrier] DMRS mask."""
        return self._pilot_mask

    @property
    def metadata(self) -> dict[str, Any]:
        """JSON-serializable configuration and implementation identity."""
        return dict(self._metadata)

    def estimate(self, received_grid: Any, noise_variance: Any) -> NRChannelEstimate:
        """Return DMRS-based LS channel estimates and interpolator error variance.

        received_grid has complex axes [batch, rx, rx antenna, OFDM symbol, FFT bin].
        noise_variance is a nonnegative AWGN variance scalar or broadcastable tensor.
        """
        torch = self._torch
        if not torch.is_tensor(received_grid):
            received_grid = torch.as_tensor(received_grid, device=self.device)
        if not torch.is_complex(received_grid):
            raise ValueError("received_grid must have a complex dtype")
        received_grid = received_grid.to(device=self.device, dtype=torch.complex64)
        expected_tail = (
            self._resource_grid.num_ofdm_symbols,
            self._resource_grid.fft_size,
        )
        if received_grid.ndim != 5 or tuple(received_grid.shape[-2:]) != expected_tail:
            raise ValueError(
                "received_grid must have [batch, rx, rx_ant, "
                f"{expected_tail[0]} symbols, {expected_tail[1]} subcarriers] shape"
            )
        if any(size <= 0 for size in received_grid.shape[:3]):
            raise ValueError("batch, receive, and receive-antenna dimensions must be positive")
        if not bool(torch.isfinite(received_grid).all()):
            raise ValueError("received_grid must be finite")

        noise = torch.as_tensor(noise_variance, dtype=torch.float32, device=self.device)
        if not bool(torch.isfinite(noise).all()) or bool((noise < 0).any()):
            raise ValueError("noise_variance must be finite and nonnegative")

        h_hat, error_variance = self._estimator(received_grid, noise)
        return NRChannelEstimate(
            h_hat=h_hat,
            error_variance=error_variance,
            pilot_mask=self.pilot_mask.to(device=self.device),
            metadata=self.metadata,
        )

    def observe(self, sample_id: str, received_grid: Any, noise_variance: Any) -> NRObservation:
        """Estimate and package receiver-visible data with physical FFT bins."""
        estimate = self.estimate(received_grid, noise_variance)
        return NRObservation.from_receiver_output(
            sample_id=sample_id,
            received_grid=received_grid,
            channel_estimate=estimate,
            effective_subcarrier_indices=self._resource_grid.effective_subcarrier_ind,
            receiver_config=self.config,
        )

    def transmit(self, batch_size: int, *, seed: int = 0) -> tuple[Any, Any]:
        """Generate PUSCH resource grids and bits reproducibly.

        This exposes the configured Sionna transmitter without leaking bits
        into :class:`NRObservation` or its inference payload.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return call_with_sionna_seed(
            self.device,
            seed,
            lambda: self._transmitter(batch_size),
        )

    def place_allocation_in_slot(self, allocation_grid: Any) -> Any:
        """Place this allocation-local frequency grid at its configured slot symbols.

        The result has a 14-symbol slot axis. Unscheduled symbols are zero-filled;
        this helper does not multiplex other physical channels into those symbols.
        """
        torch = self._torch
        grid = torch.as_tensor(allocation_grid, device=self.device)
        if not torch.is_complex(grid):
            raise ValueError("allocation_grid must have a complex dtype")
        if grid.ndim != 5:
            raise ValueError("allocation_grid must have [batch, tx, stream, symbol, fft] axes")
        expected = (self._resource_grid.num_ofdm_symbols, self._resource_grid.fft_size)
        if tuple(grid.shape[-2:]) != expected:
            raise ValueError(f"allocation_grid must end with {expected} symbol/FFT axes")
        if not bool(torch.isfinite(grid).all()):
            raise ValueError("allocation_grid must be finite")

        start_symbol, duration = self.config.symbol_allocation
        slot_grid = torch.zeros(
            (*grid.shape[:-2], 14, self._resource_grid.fft_size),
            dtype=grid.dtype,
            device=self.device,
        )
        slot_grid[..., start_symbol : start_symbol + duration, :] = grid
        return slot_grid

    def extract_allocation_from_slot(self, slot_grid: Any) -> Any:
        """Extract the configured PUSCH symbol interval from a 14-symbol slot grid."""
        torch = self._torch
        grid = torch.as_tensor(slot_grid, device=self.device)
        if not torch.is_complex(grid):
            raise ValueError("slot_grid must have a complex dtype")
        if grid.ndim != 5:
            raise ValueError("slot_grid must have [batch, rx, rx_ant, symbol, fft] axes")
        expected = (14, self._resource_grid.fft_size)
        if tuple(grid.shape[-2:]) != expected:
            raise ValueError(f"slot_grid must end with {expected} symbol/FFT axes")
        if not bool(torch.isfinite(grid).all()):
            raise ValueError("slot_grid must be finite")

        start_symbol, duration = self.config.symbol_allocation
        return grid[..., start_symbol : start_symbol + duration, :]

    def modulate_slot_grid(self, slot_grid: Any) -> Any:
        """OFDM-modulate a full-slot frequency grid with the configured Sionna CP."""
        try:
            from sionna.phy.ofdm import OFDMModulator
        except ImportError as exc:
            raise RuntimeError("Install the optional Sionna 2.2 PHY dependencies") from exc

        grid = self._torch.as_tensor(slot_grid, device=self.device)
        self.extract_allocation_from_slot(grid)  # Validate axes, dtype, and finite values.
        modulator = OFDMModulator(
            cyclic_prefix_length=self._resource_grid.cyclic_prefix_length,
            device=self.device,
        )
        return modulator(grid)

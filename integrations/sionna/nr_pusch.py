"""Optional Sionna PHY adapter for NR PUSCH DMRS channel estimation."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from receivers.nr import NRChannelEstimate, NRPUSCHConfig


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
        pusch_config.n_size_bwp = config.n_size_bwp
        pusch_config.num_antenna_ports = config.num_antenna_ports
        pusch_config.num_layers = config.num_layers
        pusch_config.dmrs.config_type = config.dmrs_config_type
        pusch_config.dmrs.length = config.dmrs_length
        pusch_config.dmrs.additional_position = config.dmrs_additional_position
        pusch_config.dmrs.num_cdm_groups_without_data = config.num_cdm_groups_without_data
        pusch_config.mapping_type = config.mapping_type
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

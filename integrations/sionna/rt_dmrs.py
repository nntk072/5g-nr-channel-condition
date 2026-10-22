"""Build NR DMRS observations from frequency-domain Sionna RT channels."""

from __future__ import annotations

import math

import numpy as np
from artifacts.provenance import package_source_sha256
from datasets.channel import ChannelDataset
from receivers.nr import NRPUSCHConfig

from .nr_pusch import SionnaPUSCHLSReceiver


def thermal_noise_variance_w(
    *, temperature_k: float, noise_figure_db: float, bandwidth_hz: float
) -> float:
    """Return complex-baseband thermal-noise power ``k T B F`` in watts."""
    for name, value in (("temperature_k", temperature_k), ("bandwidth_hz", bandwidth_hz)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(noise_figure_db) or noise_figure_db < 0:
        raise ValueError("noise_figure_db must be finite and nonnegative")
    boltzmann_j_per_k = 1.380649e-23
    noise_factor = 10.0 ** (noise_figure_db / 10.0)
    return boltzmann_j_per_k * temperature_k * bandwidth_hz * noise_factor


def resample_centered_cfr(
    channels: np.ndarray,
    *,
    source_spacing_hz: float,
    target_subcarriers: int,
    target_spacing_hz: float,
) -> np.ndarray:
    """Linearly resample centered CFRs, preserving each receive-antenna axis."""
    values = np.asarray(channels)
    if (
        values.ndim != 3
        or values.shape[1] < 1
        or values.shape[2] < 8
        or not np.iscomplexobj(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("channels must be finite complex [samples, rx_ant, >=8 tones]")
    for name, value in (
        ("source_spacing_hz", source_spacing_hz),
        ("target_spacing_hz", target_spacing_hz),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(target_subcarriers, bool)
        or not isinstance(target_subcarriers, int)
        or target_subcarriers < 8
    ):
        raise ValueError("target_subcarriers must be an integer >= 8")

    source_frequencies = (
        np.arange(values.shape[-1], dtype=np.float64) - values.shape[-1] // 2
    ) * source_spacing_hz
    target_frequencies = (
        np.arange(target_subcarriers, dtype=np.float64) - target_subcarriers // 2
    ) * target_spacing_hz
    if (
        target_frequencies[0] < source_frequencies[0]
        or target_frequencies[-1] > source_frequencies[-1]
    ):
        raise ValueError("target frequency grid must lie inside the source CFR bandwidth")

    resampled = np.empty((len(values), values.shape[1], target_subcarriers), dtype=np.complex128)
    for sample_index in range(len(values)):
        for antenna_index in range(values.shape[1]):
            channel = values[sample_index, antenna_index]
            resampled[sample_index, antenna_index] = np.interp(
                target_frequencies, source_frequencies, channel.real
            ) + 1j * np.interp(target_frequencies, source_frequencies, channel.imag)
    return resampled.astype(np.complex64)


def build_rt_dmrs_observation_dataset(
    datasets: list[ChannelDataset],
    *,
    snr_db: float | None = 12.0,
    tx_power_per_re_w: float | None = None,
    noise_figure_db: float = 7.0,
    temperature_k: float = 290.0,
    seed: int = 20261031,
    device: str = "cuda:0",
    n_size_bwp: int = 6,
    subcarrier_spacing_khz: int = 30,
) -> ChannelDataset:
    """Apply a native Sionna PUSCH/DMRS receiver to saved RT CFR samples.

    By default, each sample is normalized to unit mean channel power and receives
    AWGN at the requested per-link SNR. To preserve absolute path loss instead,
    set ``snr_db=None`` and ``tx_power_per_re_w``; this scales each active
    resource element to the supplied average transmit power and adds thermal
    noise ``k T B F`` using the subcarrier spacing as the noise bandwidth. The
    resulting LS estimate is divided by the transmit-amplitude scale so its
    values remain channel gains, with physical SNR differences retained.

    Both modes idealize timing, CFO, and inter-symbol channel variation; neither
    is a complete NR link simulation.
    """
    if not datasets:
        raise ValueError("at least one RT dataset is required")
    physical_power_mode = tx_power_per_re_w is not None
    if tx_power_per_re_w is not None:
        if snr_db is not None:
            raise ValueError("snr_db must be None when tx_power_per_re_w is set")
        if (
            isinstance(tx_power_per_re_w, bool)
            or not math.isfinite(tx_power_per_re_w)
            or tx_power_per_re_w <= 0
        ):
            raise ValueError("tx_power_per_re_w must be finite and positive")
    else:
        if snr_db is None or not math.isfinite(snr_db):
            raise ValueError("snr_db must be finite unless absolute-power mode is selected")
        noise_variance_w = None
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 2:
        raise ValueError("seed must be an integer in [0, 2**63 - 2]")
    if isinstance(n_size_bwp, bool) or not isinstance(n_size_bwp, int) or n_size_bwp < 1:
        raise ValueError("n_size_bwp must be a positive integer")
    if (
        isinstance(subcarrier_spacing_khz, bool)
        or not isinstance(subcarrier_spacing_khz, int)
        or subcarrier_spacing_khz <= 0
    ):
        raise ValueError("subcarrier_spacing_khz must be a positive integer")
    if tx_power_per_re_w is not None:
        noise_variance_w = thermal_noise_variance_w(
            temperature_k=temperature_k,
            noise_figure_db=noise_figure_db,
            bandwidth_hz=subcarrier_spacing_khz * 1_000.0,
        )

    data = ChannelDataset.concatenate(datasets)
    metadata_grid = datasets[0].metadata[0].get("grid_config", {})
    source_spacing_hz = float(metadata_grid.get("spacing_hz", 0.0))
    source_subcarriers = metadata_grid.get("n_subcarriers")
    if source_subcarriers != data.h.shape[-1]:
        raise ValueError("RT grid metadata must match the CFR tone count")
    for dataset in datasets:
        for item in dataset.metadata:
            grid = item.get("grid_config", {})
            if (
                grid.get("spacing_hz") != source_spacing_hz
                or grid.get("n_subcarriers") != source_subcarriers
            ):
                raise ValueError("all input RT datasets must use the same frequency grid")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and Sionna PHY to build NR DMRS observations") from exc
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    label_semantics = _label_semantics(data)

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
        ),
        device=device,
    )
    resource_grid = receiver.resource_grid
    if resource_grid.num_effective_subcarriers != resource_grid.fft_size:
        raise ValueError("RT DMRS adapter currently requires a full-band PUSCH resource grid")
    cfr = resample_centered_cfr(
        data.h,
        source_spacing_hz=source_spacing_hz,
        target_subcarriers=resource_grid.fft_size,
        target_spacing_hz=subcarrier_spacing_khz * 1_000.0,
    )
    if not physical_power_mode:
        channel_power = np.mean(np.abs(cfr) ** 2, axis=-1, keepdims=True)
        if np.any(channel_power <= np.finfo(np.float32).tiny):
            raise ValueError("every RT channel must have positive power on the PUSCH bandwidth")
        cfr = cfr / np.sqrt(channel_power)

    transmit_grid, _ = receiver.transmit(1, seed=seed)
    transmit_grid = transmit_grid.expand(len(data.y), *transmit_grid.shape[1:])
    transmit_amplitude = math.sqrt(tx_power_per_re_w) if tx_power_per_re_w is not None else 1.0
    transmit_grid = transmit_grid * transmit_amplitude
    channel_tensor = torch.as_tensor(cfr, dtype=torch.complex64, device=device).reshape(
        len(data.y), 1, 1, 1, resource_grid.fft_size
    )
    noiseless = transmit_grid * channel_tensor
    if noise_variance_w is None:
        assert snr_db is not None
        noise_variance = noiseless.abs().square().mean() * (10.0 ** (-snr_db / 10.0))
    else:
        noise_variance = torch.as_tensor(noise_variance_w, dtype=torch.float32, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 1)
    noise = torch.complex(
        torch.randn(noiseless.shape, dtype=torch.float32, device=device, generator=noise_generator),
        torch.randn(noiseless.shape, dtype=torch.float32, device=device, generator=noise_generator),
    ) * torch.sqrt(noise_variance / 2.0)
    with torch.no_grad():
        estimate = receiver.estimate(noiseless + noise, noise_variance)
    observed_cfr = estimate.h_hat.mean(dim=5)[:, 0, 0, 0, 0, :].detach().cpu().numpy()
    if tx_power_per_re_w is not None:
        observed_cfr /= transmit_amplitude

    receiver_metadata = receiver.metadata
    observation_metadata = tuple(
        {
            **item,
            "receiver_observation": "Sionna PUSCH DMRS LS channel estimate",
            "receiver_snr_db": snr_db,
            "tx_power_per_re_w": tx_power_per_re_w,
            "noise_variance_w": noise_variance_w,
            "receiver_ideal_synchronization": True,
        }
        for item in data.metadata
    )
    return ChannelDataset(
        h=observed_cfr[:, None, :].astype(np.complex64),
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=observation_metadata,
        label_semantics=label_semantics,
        generation_report={
            "schema": "loslab.rt_dmrs_observation.v1",
            "label_semantics": label_semantics,
            "observation_semantics": "Sionna PUSCH DMRS LS estimate; ideal synchronization",
            "receiver_metadata": receiver_metadata,
            "input_grid": {
                "n_subcarriers": int(source_subcarriers),
                "spacing_hz": source_spacing_hz,
            },
            "resampling": "linear interpolation of centered complex CFR onto full PUSCH grid",
            "power_mode": (
                "absolute_tx_power_thermal_noise"
                if physical_power_mode
                else "normalized_per_link_snr"
            ),
            "normalization": (
                "none; LS output divided by sqrt(tx_power_per_re_w)"
                if physical_power_mode
                else "unit mean channel power per link before AWGN"
            ),
            "snr_db": snr_db,
            "tx_power_per_re_w": tx_power_per_re_w,
            "noise_figure_db": noise_figure_db if physical_power_mode else None,
            "temperature_k": temperature_k if physical_power_mode else None,
            "noise_bandwidth_hz": subcarrier_spacing_khz * 1_000.0 if physical_power_mode else None,
            "noise_variance_w": noise_variance_w,
            "noise_variance_semantics": (
                "k_B * temperature_k * noise_bandwidth_hz * 10**(noise_figure_db/10)"
                if physical_power_mode
                else "per-link signal power divided by requested linear SNR"
            ),
            "seed": seed,
            "noise_seed": seed + 1,
            "sample_count": len(data.y),
            "channel_condition_label_semantics": label_semantics,
            "limitations": [
                "No timing/CFO error, inter-symbol channel variation, or CP violation is simulated.",
                "RT labels are geometric depth-zero direct-path oracles, not measured LOS/NLOS labels.",
                *(
                    []
                    if physical_power_mode
                    else ["Per-link power normalization removes path-loss information."]
                ),
            ],
            "package_source_sha256": package_source_sha256(),
        },
    )


def _label_semantics(data: ChannelDataset) -> str:
    """Resolve one canonical label meaning and reject ambiguous input data."""
    aliases = {
        "GEOMETRIC_DIRECT_PATH_ORACLE": "geometric_direct_path_oracle",
        "PROFILE_PROXY": "profile_proxy",
        "SYNTHETIC_DEBUG_CONSTRUCTED": "synthetic_debug_direct_component",
        "SYNTHETIC_DEBUG_DIRECT_COMPONENT": "synthetic_debug_direct_component",
        "UNSPECIFIED": "unspecified",
        "UNKNOWN": "unspecified",
        "UNKNOWN_LEGACY": "unspecified",
    }
    declared = data.label_semantics
    values: set[str] = set()
    has_unresolved = False
    for metadata in data.metadata:
        raw = metadata.get("geometry_label") or metadata.get("label_semantics")
        if raw is None:
            if declared == "unspecified_legacy":
                has_unresolved = True
            else:
                values.add(declared)
            continue
        normalized = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
        semantic = aliases.get(normalized, normalized.lower())
        if semantic == "unspecified":
            has_unresolved = True
        else:
            values.add(semantic)
    if declared != "unspecified_legacy" and values and values != {declared}:
        raise ValueError("RT dataset label metadata conflicts with its declared semantics")
    if has_unresolved or len(values) != 1:
        raise ValueError("RT datasets need one explicit, consistent label semantic")
    return next(iter(values))

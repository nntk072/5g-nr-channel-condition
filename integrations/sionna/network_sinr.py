"""Bounded, replay-identifiable resource-grid network SINR workflow."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .network_phy import (
    NetworkGridComposition,
    NetworkLinkBudget,
    NetworkResourceSINR,
    NetworkSINR,
    apply_network_link_budget,
    compose_downlink_resource_grids,
    network_resource_sinr,
    network_sinr,
)


@dataclass(frozen=True, slots=True)
class NetworkGridSINRWorkflowResult:
    """Grid composition, resource SINR, and source-bound software report."""

    composition: NetworkGridComposition
    metrics: NetworkResourceSINR
    aggregate_metrics: NetworkSINR | None
    link_budget: NetworkLinkBudget | None
    report: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha256(torch: Any, value: Any) -> str:
    array = value.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": [int(dim) for dim in array.shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_network_grid_sinr(
    channel: Any,
    transmit_grids: Any,
    *,
    transmit_power_w_per_re: Any,
    serving_tx_indices: Any,
    noise_variance_w_per_re: Any,
    link_budget_parameters: dict[str, Any] | None = None,
    max_tensor_elements: int = 16_000_000,
) -> NetworkGridSINRWorkflowResult:
    """Compose an aligned network resource grid and report per-RE SINR.

    The channel is dimensionless complex frequency-domain gain with axes
    ``[batch, rx, rx_port, tx, tx_port, symbol, subcarrier]``. This synchronized,
    same-numerology workflow is a software diagnostic; it does not model RF
    acquisition, asynchronous cells, or qualify a receiver.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch to run the network grid SINR workflow") from error

    if not torch.is_tensor(channel) or not torch.is_complex(channel) or channel.ndim != 7:
        raise TypeError("channel must be a rank-7 complex PyTorch tensor")
    if (
        not torch.is_tensor(transmit_grids)
        or not torch.is_complex(transmit_grids)
        or transmit_grids.ndim != 5
    ):
        raise TypeError("transmit_grids must be a rank-5 complex PyTorch tensor")
    if (
        isinstance(max_tensor_elements, bool)
        or not isinstance(max_tensor_elements, int)
        or max_tensor_elements < 1
    ):
        raise ValueError("max_tensor_elements must be a positive integer")
    if tuple(channel.shape[0:1]) != tuple(transmit_grids.shape[0:1]):
        raise ValueError("channel and transmit grid batch dimensions differ")
    if tuple(channel.shape[3:]) != tuple(transmit_grids.shape[1:]):
        raise ValueError("channel TX-port/symbol/subcarrier axes do not match transmit grids")
    if any(int(size) <= 0 for size in (*channel.shape, *transmit_grids.shape)):
        raise ValueError("channel and transmit grid axes must be non-empty")

    batch, rx_count, rx_ports, tx_count, _tx_ports, symbols, subcarriers = map(int, channel.shape)
    if link_budget_parameters is not None:
        if not isinstance(link_budget_parameters, dict):
            raise TypeError("link_budget_parameters must be a dictionary when provided")
        required = {
            "path_loss_db",
            "tx_antenna_gain_dbi",
            "rx_antenna_gain_dbi",
            "tx_beam_gain_dbi",
            "rx_beam_gain_dbi",
            "input_channel_semantics",
        }
        allowed = required | {"subcarrier_frequencies_hz"}
        if set(link_budget_parameters) - allowed or not required.issubset(link_budget_parameters):
            raise ValueError(
                "link_budget_parameters must declare all five profiles and channel semantics"
            )
        if link_budget_parameters["input_channel_semantics"] != "unit_mean_small_scale_fading":
            raise ValueError("link budget requires unit_mean_small_scale_fading channel semantics")

    input_cells = channel.numel() + transmit_grids.numel()
    contribution_cells = batch * rx_count * rx_ports * tx_count * symbols * subcarriers
    received_cells = batch * rx_count * rx_ports * symbols * subcarriers
    metric_cells = rx_count * symbols * subcarriers * 5
    # Reserve two contribution-sized buffers for einsum and source validation,
    # plus the scaled grid, received grid, and five returned metric arrays.
    budget_cells = channel.numel() if link_budget_parameters is not None else 0
    estimated_live_cells = (
        input_cells + budget_cells + 2 * contribution_cells + received_cells + metric_cells
    )
    if estimated_live_cells > max_tensor_elements:
        raise ValueError(
            "network grid SINR workflow exceeds max_tensor_elements before composition"
        )

    link_budget = None
    effective_channel = channel
    if link_budget_parameters is not None:
        link_budget = apply_network_link_budget(
            channel,
            transmit_power_w_per_re=transmit_power_w_per_re,
            **link_budget_parameters,
        )
        effective_channel = link_budget.channel
    composition = compose_downlink_resource_grids(
        effective_channel, transmit_grids, transmit_power_w_per_re
    )
    metrics = network_resource_sinr(composition, serving_tx_indices, noise_variance_w_per_re)
    serving_probe = torch.as_tensor(serving_tx_indices, device=channel.device)
    aggregate_metrics = None
    if tuple(serving_probe.shape) == (rx_count,):
        aggregate_metrics = network_sinr(
            composition,
            serving_probe.to(dtype=torch.int64).tolist(),
            noise_variance_w_per_re,
        )

    implementation_functions: list[tuple[str, Any]] = [
        ("grid_composition", compose_downlink_resource_grids),
        ("resource_sinr", network_resource_sinr),
        ("workflow", run_network_grid_sinr),
    ]
    if link_budget is not None:
        implementation_functions.append(("link_budget", apply_network_link_budget))
    if aggregate_metrics is not None:
        implementation_functions.append(("network_sinr", network_sinr))
    implementation_sha256 = {
        name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        for name, function in implementation_functions
    }
    tensor_sha256 = {
        name: _tensor_sha256(torch, value)
        for name, value in {
            "channel": channel,
            "effective_channel": effective_channel,
            "transmit_grids": transmit_grids,
            "transmit_power_w_per_re": composition.transmit_power_w_per_re,
            "source_contributions": composition.source_contributions,
            "received_grid": composition.received_grid,
            "desired_power_w_per_re": metrics.desired_power_w_per_re,
            "interference_power_w_per_re": metrics.interference_power_w_per_re,
            "noise_power_w_per_re": metrics.noise_power_w_per_re,
            "sinr_linear": metrics.sinr_linear,
            "valid": metrics.valid,
        }.items()
    }
    serving = torch.as_tensor(serving_tx_indices, device=channel.device)
    noise = torch.as_tensor(noise_variance_w_per_re, device=channel.device)
    report: dict[str, Any] = {
        "schema": "umcd.network_grid_sinr_workflow.v1",
        "status": "OBSERVED_NOT_QUALIFIED",
        "boundary": "SYNCHRONIZED_GRID_SOFTWARE_DIAGNOSTIC_NOT_RECEIVER_QUALIFICATION",
        "dimensions": {
            "batch": batch,
            "rx": rx_count,
            "rx_port": rx_ports,
            "tx": tx_count,
            "symbol": symbols,
            "subcarrier": subcarriers,
        },
        "input_channel_semantics": (
            "unit_mean_small_scale_fading_with_declared_profiles"
            if link_budget is not None
            else "caller_supplied_effective_channel"
        ),
        "max_tensor_elements": max_tensor_elements,
        "estimated_live_tensor_elements": estimated_live_cells,
        "serving_tx_indices_sha256": _tensor_sha256(torch, serving),
        "noise_variance_input_sha256": _tensor_sha256(torch, noise),
        "implementation_sha256": implementation_sha256,
        "tensor_sha256": tensor_sha256,
        "aggregate_metrics_sha256": (
            {
                name: _tensor_sha256(torch, value)
                for name, value in {
                    "desired_power_w_per_re": aggregate_metrics.desired_power_w_per_re,
                    "interference_power_w_per_re": aggregate_metrics.interference_power_w_per_re,
                    "noise_power_w_per_re": aggregate_metrics.noise_power_w_per_re,
                    "sinr_linear": aggregate_metrics.sinr_linear,
                    "valid": aggregate_metrics.valid,
                }.items()
            }
            if aggregate_metrics is not None
            else None
        ),
        "link_budget_sha256": (
            _canonical_sha256(
                {
                    name: _tensor_sha256(torch, getattr(link_budget, name))
                    for name in (
                        "path_loss_db",
                        "tx_antenna_gain_dbi",
                        "rx_antenna_gain_dbi",
                        "tx_beam_gain_dbi",
                        "rx_beam_gain_dbi",
                        "net_gain_db",
                        "power_gain_linear",
                        "transmit_power_w_per_re",
                        "mean_received_power_w_per_re",
                    )
                }
                | {
                    "input_channel_semantics": link_budget.input_channel_semantics,
                    "subcarrier_frequencies_hz": (
                        _tensor_sha256(torch, link_budget.subcarrier_frequencies_hz)
                        if link_budget.subcarrier_frequencies_hz is not None
                        else None
                    ),
                }
            )
            if link_budget is not None
            else None
        ),
    }
    schema_path = Path(__file__).resolve().parents[2] / (
        "datasets/schema/network-grid-sinr-workflow-report.v1.schema.json"
    )
    schema_bytes = schema_path.read_bytes()
    report["schema_sha256"] = hashlib.sha256(schema_bytes).hexdigest()
    import jsonschema

    schema = json.loads(schema_bytes)
    jsonschema.Draft202012Validator.check_schema(schema)
    report["report_sha256"] = _canonical_sha256(report)
    jsonschema.Draft202012Validator(schema).validate(report)
    return NetworkGridSINRWorkflowResult(
        composition, metrics, aggregate_metrics, link_budget, report
    )

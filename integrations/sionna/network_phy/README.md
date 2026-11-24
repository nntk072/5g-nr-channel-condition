# integrations.sionna.network_phy

Sionna network PHY: records, composition, and link operations.

## Purpose

This package is one module split into concept modules. It keeps the public names of the original module, so callers import from `integrations.sionna.network_phy` unchanged.

## Modules

| Module | Contents |
| --- | --- |
| `operations.py` | Network link budget, SINR, and waveform composition operations |
| `records.py` | Network-level channel records, validators, and waveform propagation |

## Public API (26 names)

Re-exported from `__init__.py` through `__all__`:

`NetworkGridComposition`, `NetworkLinkBudget`, `NetworkMIMOChannelPropagation`, `NetworkResourceSINR`, `NetworkSINR`, `NetworkTransmitterWaveform`, `NetworkWaveformComposition`, `NoisyNetworkGrid`, `NoisyNetworkWaveform`, `_NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS`, `_contains_boolean`, `_fractional_delay_capture_slice`, `_mean_complex_power`, `_shared_phase_noise_processes`, `_validate_composition`, `_validated_nonnegative_vector`, `_validated_real_values`, `propagate_network_transmit_waveforms`, `add_independent_receiver_awgn`, `add_independent_receiver_waveform_awgn`, `apply_network_link_budget`, `apply_network_transmitter_iq_imbalance`, `compose_downlink_resource_grids`, `compose_received_waveforms`, `network_resource_sinr`, `network_sinr`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.


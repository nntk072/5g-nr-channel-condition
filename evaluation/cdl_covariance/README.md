# evaluation.cdl_covariance

Independent CDL space-time covariance references, one module per topology family.
Each module computes the reference for its frozen protocol and is checked against
the native Sionna adapter by the matching native test.

## Modules

| Module | Topology family |
| --- | --- |
| `shared.py` | Frozen reference loading and direction helpers that were identical across families |
| `spatiotemporal.py` | Single-link spatiotemporal covariance |
| `temporal_reference.py` | Temporal reference for the CDL-D LOS diffuse case |
| `joint_tx_rx.py` | Joint TX and RX spatial covariance |
| `joint_tx_rx_dual_pol.py` | Joint TX and RX with dual polarization |
| `joint_tx_rx_dual_tx_rx.py` | Dual TX and RX arrays |
| `rotated_dual_tx_rx.py` | Rotated dual TX and RX arrays |
| `patterned_rotated_dual_tx_rx.py` | Patterned, rotated dual TX and RX arrays |
| `handheld_ut.py` | Handheld user-terminal topology |
| `four_element.py` | Four-element spatial covariance |

## Invariants

- The shared module holds only helper bodies that were byte-identical in every
  family that used them. Where a family's body differs, that family keeps its own
  helper, so each reference stays independent of the others.
- File-relative paths use `parents[2]` for the repository root.

# G5 receiver support matrix (2026-11-12)

This matrix separates implemented software paths from supported qualification
scope. A runnable adapter or passing synthetic/native software regression does
not qualify receiver performance. All G5 criteria and receipts remain
`NOT_QUALIFIED`; the matrix does not authorize a receiver campaign, production
RT generation, or advanced training.

| Receiver path | Implemented boundary | Explicit unsupported or unverified boundary | Source of record |
|---|---|---|---|
| PUSCH frequency-grid estimation | Sionna PUSCH resource-grid construction, DMRS LS estimation, interpolator variance, typed NR observation, bounded decoder/CRC route. Config pins BWP, numerology, ports/layers, DMRS, mapping and allocation. | `frequency_hopping` must be `neither`; transform precoding must be false. The pinned Sionna PHY 2.2.0 receiver does not implement those modes. Neither general channel estimation accuracy nor receiver performance is qualified. | `receivers/nr/pusch.py`, `integrations/sionna/nr_pusch.py`, `PKG06-003` |
| Intra-slot hopping grid relocation | Caller-resolved hop segments can be composed into a carrier grid; a deterministic received-grid dehop helper rehomes both segments to the first-hop allocation and replays exact input/output and hop-layout digests. `SionnaPUSCHLSReceiver.decode_intra_slot_hopped_carrier_grid()` now checks that layout against its receiver configuration, verifies the relocation, then calls the ordinary non-hopping LS decoder. Its strict report binds receiver config, replay metadata, normalized noise variance, and decoded outputs. A CPU loopback moves the second half of a generated allocation to a distinct RB range and decodes the original bits/CRC after replay. | The loopback verifies software relocation and adapter plumbing only. It does not resolve grants, generate an independently standards-derived hopped DMRS grid, implement transform precoding or general backend hopping, or establish performance. The report schema and status remain `NOT_QUALIFIED`; `PKG06-003` remains partial. | `receivers/nr/pusch.py`, `integrations/sionna/nr_pusch.py`, `datasets/schema/nr-pusch-intra-slot-dehopped-grid.v1.schema.json`, `datasets/schema/nr-pusch-dehopped-grid-replay.v1.schema.json`, `datasets/schema/nr-pusch-intra-slot-hopped-decode.v1.schema.json`, `tests/test_receivers/test_nr_pusch.py`, `tests/test_integrations/test_nr_pusch_native.py` |
| Sample-domain single-link PUSCH | Standardized TDL waveform path, noise/impairment composition, DMRS receiver and decoded payload/CRC for bounded fixtures. | Current results are finite synthetic configurations. CP exceedance, acquisition error, mobility, and impairment interactions are not qualified over a general operating envelope. | `integrations/sionna/standardized_pusch.py`, `experiments/ordered_impairments_receiver.py`, `PKG06-003`, `PKG06-006` |
| Ordered impairments | Seeded AWGN, CFO, fractional timing, node-shared phase noise, and receiver I/Q imbalance have bounded software and selected decoder integrations. | Calibrated impairments, independent clock behavior across numerologies, broad interactions, and receiver qualification remain open. | `integrations/sionna/ofdm_impairments.py`, `integrations/sionna/network_phy.py`, `PKG06-004`, `PKG06-006` |
| Multi-link/multi-cell waveform | `run_network_pusch_receiver()` composes already channelized links; `run_network_pusch_from_transmitters()` applies per-node TX IQ before caller-declared full causal MIMO FIR propagation, retains FIR tails, and then runs receiver composition, seeded receiver-node AWGN and bounded PUSCH decode. The strict nested reports bind transmitter/profile/FIR/link samples, receiver inputs and outputs, implementation hashes and allocation estimates. The non-native network receiver/PHY slice passes 116 tests with one native Sionna test deselected. `run_network_grid_sinr()` also binds per-resource desired/interference/noise/SINR and can apply declared profiles to unit-mean small-scale grids. | `OBSERVED_NOT_QUALIFIED`; FIRs, grid profiles and calibration remain caller-owned. No RT channel is derived, and the network SINR grid route is synchronized and same-numerology only. Asynchronous multi-numerology, complete contamination truth, acquisition, external G5 acceptance, and G3 admission remain open. No native qualification or D1 authorization follows. | `integrations/sionna/network_receiver.py`, `integrations/sionna/network_phy.py`, `integrations/sionna/network_sinr.py`, `datasets/schema/network-pusch-workflow-report.v1.schema.json`, `datasets/schema/network-transmitter-pusch-workflow-report.v1.schema.json`, `datasets/schema/network-grid-sinr-workflow-report.v1.schema.json`, `PKG06-004` |
| SSB/PBCH | Bounded PSS/SSS acquisition, supported candidate timing, PBCH decode, typed MIB recovery, and finite pre-search allocation controls. | General ASN.1/PER extension decoding, exhaustive band/case/candidate/frame coverage, calibrated RF scaling, mobile/multi-antenna receiver performance, and qualification remain open. | `integrations/sionna/pbch_ofdm_receiver.py`, `PKG06-025` |
| PRS | Supported TS 38.211 resource mapping and a bounded frequency-domain delay/common-CFO search over known mapped PRS symbols. | This is not raw-sample acquisition or first-arrival ToA. Intra-symbol ICI, ambiguity resolution, realistic synchronization, independent conformance, and qualification remain open. | `integrations/sionna/nr_prs.py`, `PKG06-027` |
| SRS positioning | Sequence/resource mapping, decoded-state scheduling, bounded grid measurement, and report-arithmetic contracts. | Raw RRC/PER and CRC/polar configuration, first-path and clock calibration, full physical scheduling, synchronized multiport receiver evidence, and qualification remain open. | `integrations/sionna/srs_measurements.py`, `PKG06-010` |

## Admission contract for a bounded receiver study

Before any G5 qualification run, freeze the exact waveform, receiver and
decoder configuration, supported channel/impairment strata, independent
channel-cluster unit, SNR grid, transport-block roster, failure/outcome
accounting, and task-specific acceptance thresholds in the signed preregistered
contract. The example thresholds in
[`SNR_FAMILY_ACCEPTANCE_CONTRACT_V1.md`](SNR_FAMILY_ACCEPTANCE_CONTRACT_V1.md)
are not defaults. The executable cluster floor is not a power guarantee.
Trusted time, signer/keyring custody, and independent reviewer authority must
be established separately. Preserve failed outcomes and unresolved grid
crossings; do not run extrapolated SNR@10% comparisons.

G3 receiver-impact evaluation consumes outcomes from a separately G5-qualified
PUSCH receiver and compares candidate/challenger configurations on the frozen
common SNR grid. Its numerical scorer is not evidence that the upstream
receiver, channel, or outcomes were qualified. G3's task-specific complex-CSI,
delay/angular, path-equivalence, and receiver-impact criteria remain defined in
the Unified Multipath Superdataset plan; each actual run still requires an
immutable task-specific contract and powered sample-size design.

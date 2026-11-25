# receivers

**Purpose.** Receiver-side observations and reference receiver processing for
NR uplink (PUSCH) and for pilot-based channel observations.

**Public API.**

- `receivers.nr.pusch` — framework-independent contracts for NR PUSCH receiver
  observations.
- `receivers.nr.pusch_core` — intra-slot frequency-hopping DM-RS plans,
  channel estimation, demapping, and dehopping (`constants`, `dmrs`,
  `estimation`, `hopping`, `mapping`, `types`).
- `receivers.pilots` — nonideal pilot-based CFR observation.

**Invariants.**

- `receivers.pilots` is **not** a standards-compliant NR DM-RS receiver. Its
  output is a research observation and must not be labelled as NR-compliant.
- Grid sizes and hopping parameters are validated against the carrier-grid
  limits in `pusch_core.constants` before any array is allocated.
- The core is independent of Sionna. Sionna-backed receivers live in
  `integrations.sionna`.

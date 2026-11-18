# Canonical target producer and consumer matrix

**Status:** repository-source audit, partial implementation evidence; not
scientific qualification. The machine-readable row inventory is
[TARGET_FAMILY_PRODUCER_CONSUMER_MATRIX.json](TARGET_FAMILY_PRODUCER_CONSUMER_MATRIX.json).

The inventory covers all 19 recognized target-definition versions in
`datasets.targets._CANONICAL_TARGET_CONTRACTS` and its provenance registry.
Each row names the current or historical producer, stored shape/axis policy,
publication/read policy, serialization contracts, and read/admission routes.
The versioned schema is
[`datasets/schema/target-family-producer-consumer-matrix.v1.schema.json`](../../datasets/schema/target-family-producer-consumer-matrix.v1.schema.json).
The regression in `tests/test_datasets/test_target_contracts.py` validates that
schema and fails when a recognized definition is missing from the inventory,
is duplicated, or cites a missing source/schema file.

The source audit distinguishes the following states:

- Current axis-bound UMCD targets: standardized and TDL-PUSCH CFR v2, RT
  absolute and symbol CFR v2, dynamic geometric visibility v4, and RMS delay
  spread v2.
- Current fixed-shape producer outputs: standardized profile proxy v1 in
  schema-v5 NR records, standardized profile proxy v2 in the standardized-run
  store, static visibility v2 in UMCD, and per-record RT geometric visibility
  v2 in NR. The profile targets remain explicit profile-category proxies,
  distinct from geometric LOS.
- Superseded or axis-ambiguous definitions remain readable but cannot be
  published through the canonical UMCD/NR writers where a producer-backed
  successor exists. The inventory records historical geometric-visibility
  definitions separately rather than upgrading their axes or rank on read.
- The inventory does not invent producers for detectable direct path,
  relative first-path delay, receiver-estimated delay spread, or physical
  received-path power. Their detector, timing, receiver, or calibrated
  link-budget references are not frozen.

## Current target admission audit

This table records the software checks found at the producer/storage boundary.
It is deliberately separate from truth qualification:

| Current family | Enforced software invariant | Remaining evidence boundary |
| --- | --- | --- |
| Standardized profile proxies v1/v2 | Values are binary; canonical fixed-singleton forms use axis-free `(1,)` storage. The semantic is a profile category, not geometric LOS. | This does not establish outdoor LOS or measured channel truth. |
| Static geometric visibility v2 | Binary value and fixed-singleton shape; target entity must be an episode link. | Storage does not re-run the independent scene visibility oracle or authenticate release-specific geometry provenance. |
| Dynamic geometric visibility v4 | Binary values, digest-bound `frame` axis, exact episode frame roster, and link entity. | The geometry cross-check and scene-asset identity require their separate G1 evidence. |
| NR trajectory geometric visibility v2 | Binary fixed-singleton value for a timestamp record; axis-ambiguous UMCD publication is read-only. | The scalar is not upgraded to a frame array, and source geometry qualification remains separate. |
| Standardized, TDL-PUSCH, and RT CFR v2 | Finite complex values; masked storage is zero with an explicit reason; definition axes are version-bound; UMCD effective-subcarrier extents match the parent grid and NR axes match the observation/FFT contract. | There is no uncalibrated coefficient magnitude cap. Native path/CFR parity, phase/power calibration, and receiver impact remain G2/G3 evidence. |
| Path-oracle RMS delay spread v2 | Finite nonnegative seconds; digest-bound frame axis and exact roster; bundle admission requires and replays the same path table, including values, mask, reasons, censoring, and the single-port rule. `NO_PATH`/zero-power are unavailable; solver failure rejects. | This checks internal source consistency only. Multiport aggregation and physical path/channel truth are not qualified. |
| Superseded/axis-ambiguous versions | Historical values and ranks remain readable; canonical publication is blocked where a reviewed successor exists. | Historical reads never infer axes or scientific meaning. |

No additional producer-backed range, gauge, or missingness invariant was
identified in the current-version source audit beyond those listed above.
Families whose detector, timing, finite-bandwidth receiver, or calibrated
power references are not frozen remain unimplemented by design.

**Verification:** the inventory parity regression validates the versioned
schema and complete version set; JSON parsing and `git diff --check` pass. This
source map is supplemented by producer-backed NR/UMCD store round trips and a
10-case digest-valid historical UMCD readback across all write-protected
versions plus RMS v1. The current writer-consumer software audit does not
complete target families whose physical references or independent geometry
evidence remain unavailable.

# G4.10 physical dual-cross HandheldUTArray CDL spatiotemporal covariance protocol — 2026-11-10

## Scientific claim

This preregistration freezes a finite Sionna PHY 2.2.0 CPU implementation-consistency check of the independent TR 38.901 V19.2.0 CDL cross-time covariance oracle for Sionna's physical dual-polarized HandheldUTArray. The observable is E[h_i(0)^* h_j(tau)] for all ordered physical UT port pairs. The reference computes fields, port phases, Doppler, Jones/XPR moments, and CDL-D/E specular components analytically without using native Sionna channel samples to form the target.

A pass is limited to the exact geometry and operating point below; it is not independent physical validation of the 3GPP specification or a release qualification.

## Frozen family and implementation

- CDL profiles: A, B, C, D, E; Sionna profile tables pinned to release 19.2.
- UT: physical Sionna HandheldUTArray; dual polarization; polarization type cross; pattern 38.901-handheld. The first polarization uses the port-local theta field; the second uses the port-local phi field; candidate positions are duplicated in polarization-grouped order.
- Antenna roster tr38901-4: candidate positions (1, 7, 3, 5), four locations, eight ports. Roster tr38901: all eight standardized candidate positions, sixteen ports.
- Handset dimensions: 0.15 m depth x 0.07 m width, with native physical per-port boresights and pi/4 polarization basis rotation for cross.
- UT orientation (0.31, -0.22, 0.55) radians; velocity (-8, 22, 17) m/s.
- Carrier 28 GHz (FR2); time lag 0.2 ms. The native channel has two time steps and sampling frequency 1/lag.
- BS: one single-V omni PanelArray port; orientation (0, 0, 0) radians.
- Channel generation: delay spread 100 ns, 30 kHz OFDM subcarrier spacing, eight subcarriers, CPU.
- Independent stochastic channel draws: **32,768 per case**, exact.
- Cases: five profiles x two rosters = **10 cases**.
- Claims: 5 x (8x8 + 16x16) x 2 = **3,200 real/imaginary coordinates**; every coordinate retained with estimate, reference, absolute error, simultaneous interval half-width, and pass flag.
- Family-wise alpha: 0.01. Use a two-sided Bonferroni standard-normal interval across all 3,200 entries with sample standard error computed across independent channel draws.
- An entry passes if |estimate-reference| <= Bonferroni half-width **and** half-width <= 0.05 native normalized-power units; an entire case passes only if all its entries pass.
- Formal seed roster: **2,821,000–2,821,009 inclusive**, assigned profile-major A, B, C, D, E and roster order tr38901-4, tr38901. The formal runner collects *all* entries even if a case fails.
- Once frozen, no changes to seeds, sample sizes, observable, family, confidence rule, precision threshold, runtime source pins, or analytical reference are allowed to reinterpret a formal result. Frozen failure remains failure.

## Pinned reference and source identities

- Frozen ETSI-derived profile/tap/angle fixture: docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json, SHA-256 24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090.
- ETSI source bytes SHA-256 b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4.
- Sionna antenna.py SHA-256 09e20b89f4c4d990bed1a3ce3d6314f9fe7e45c090737a6f5790166d4a63f926.
- Sionna channel_coefficients.py SHA-256 47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc.
- Sionna cdl.py SHA-256 91e90eb7498a3725f628f2f4d4dca6c8d7438ac549bd5d68a3f45563f282361d.
- The statistical native test also pins the five Sionna profile-table digests. The runner retains SHA-256 digests of its oracle, adapter, tests, preregistration, audit, reference, and itself.

## Independent oracle checks and excluded exploratory results

Before the formal run, native deterministic geometry and field checks compare independent dual-port positions, local bases, polarization component selection, and rotated theta/phi fields for both rosters and both VH and cross configurations. Statistical qualification here applies **only to cross**.

Two **non-admissible, pre-freeze exploratory** sizing diagnostics used distinct seeds and must never be pooled with or substituted for formal evidence:

1. Ten-case 4,096-draw pilot on seeds 9,821,000–9,821,009: 202 precision/interval failures among 3,200 coordinates; worst absolute error 0.05770260570842142; worst half-width 0.11640576105348399. Failures per A/B/C/D/E across 4- and 8-location rosters were (24,59), (12,41), (10,49), (1,6), and (0,0).
2. One 32,768-draw sample-size diagnostic, CDL-A full sixteen-port roster, seed 9,822,000: zero failures, worst absolute error 0.010699746082003855, worst simultaneous half-width 0.03962912129586528. This was only a sizing check, not statistical confirmation for the ten formal cases.

The 32,768-draw count was selected before any formal-seed observation. The scope does not change in response to formal outcomes.

## Seed audit and admissibility

The pre-run seed roster was audited by scripts/audit_seed_roster_collisions.py across the saved .json and .json.gz corpus in docs/research/results, outputs, and artifacts. Audit output:
docs/research/G4_CDL_PHYSICAL_DUAL_CROSS_HANDHELD_UT_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json.

It scanned 2,301 files; no reserved-seed collisions, malformed seed fields, or incomplete files were reported. SHA-256 of retained audit JSON: 0f4b7953040d7fec46524a8ba18c139113828af425225081ceaee33b6e3310fa. This is **corpus-scoped only**; global seed exclusivity is not established.

Formal execution requires a **clean committed working tree**, the complete audit roster, and pinned sources. The runner refuses to overwrite an existing result or checksum, and accepts native pytest exit code 1 only when every frozen statistical claim and failure has been collected and verified. Any infrastructure error or incomplete collection is not admissible evidence. A fully passing family is PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE; a completed family with failed claims is FROZEN_FAILURE_WITHIN_BOUNDED_PROTOCOL_SCOPE.

## Interpretation limits and release gates

This experiment does not establish arbitrary dual-pol handheld designs, VH statistical qualification, arbitrary candidate subsets or handset dimensions, validity at other carrier frequencies or in FR1, continuously varying orientation/velocity/lag, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, general 3GPP conformance, or overall G4 qualification.

The historical frozen 159/160 joint TX/RX 2x2 covariance failure remains retained unchanged. Production D1 generation, canonical retraining, and advanced training remain gated by the broader physical-validation and data-release criteria, regardless of this result.

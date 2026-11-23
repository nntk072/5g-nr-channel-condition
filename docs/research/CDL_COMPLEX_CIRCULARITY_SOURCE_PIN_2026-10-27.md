# G4 CDL diffuse complex circularity source pin

## Scope

The existing CDL circularity regression checks diffuse cluster coefficients for proper-complex behavior. For the first cluster in LOS CDL-D/E, the test temporarily replaces Sionna's private `_step_11` method so it can inspect the diffuse term before Sionna combines the deterministic LOS ray. The regression remains a bounded Sionna 2.2.0 software check and does not establish full G4 statistical or physical-channel qualification.

## Audited implementation source

The separate source checkout at commit `2f0a3820269b53fd022780a6f2f5f720f7ceb9b4` was compared with the installed Sionna 2.2.0 source used by the native tests. Both exact source files match:

| File | SHA-256 |
| --- | --- |
| `sionna/phy/channel/tr38901/cdl.py` | `91e90eb7498a3725f628f2f4d4dca6c8d7438ac549bd5d68a3f45563f282361d` |
| `sionna/phy/channel/tr38901/channel_coefficients.py` | `47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc` |

The test now hashes the loaded implementation files before applying the private-method override. A matching version string with changed source bytes is rejected. The pinned source digests bind the mechanics of this decomposition; they do not make the private API stable or substitute for a public diffuse-channel interface.

## Existing statistical contract

The regression's existing design is unchanged: 32,768 independent channel draws/profile; CDL-A–E diffuse cluster properness; familywise alpha 0.01; simultaneous component intervals; maximum interval half-width 0.05. The expected profile roster is separately bound to the ETSI-derived TR 38.901 V19.2.0 tap/angle fixture (`24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`), whose pinned source PDF SHA-256 is `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.

The source-pin regression remains unchanged. On 2026-11-09 the pre-existing 32,768-draw native statistical family was executed without changing seeds, family-wise alpha, claim accounting, or the 0.05 precision cap. A fresh audit clears seeds 933165–933169 within 2,285 saved JSON/JSON.GZ artifacts. The exact module passes 8/8 tests. Claim-level retention covers all 194 real/imaginary pseudocovariance coordinates across CDL-A–E, including the source-pinned diffuse-only first-cluster decomposition for LOS CDL-D/E; all 194 pass. The largest simultaneous half-width is 0.02237055 and the largest |estimate| + half-width is 0.03569447. Result: docs/research/results/g4-cdl-complex-circularity-native-validation-20261109.json (SHA-256 564e474b375e51d7f09d41253ad8a2bcda83ea83b9ab065f71a76f804722b11e); seed audit: docs/research/G4_CDL_COMPLEX_CIRCULARITY_SEED_AUDIT_2026-11-09.json. Overall G4 remains open.

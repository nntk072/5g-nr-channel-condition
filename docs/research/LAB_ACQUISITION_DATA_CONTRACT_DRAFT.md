# G3.03 Lab Acquisition Data Contract (Draft)

**Status:** planning checklist only. This draft does not change the frozen G3 acceptance criteria, authorize a qualification run, or open production RT generation. It contains no measurement data.

## Purpose and boundary

This is the minimum evidence package for future material-specific, angle-resolved measurements at the fixed **3.5 GHz** target carrier. It should let an independent reviewer reconstruct each calibrated observation, bind scored rows to immutable source files and physical acquisition units, and verify the preregistered holdout.

The current evaluator consumes summary measurements and predictions. It does not ingest or validate raw/calibrated instrument files, calibration records, specimen identity, instrument traceability, acquisition chronology, physical independence, or legal rights. Those require a separate acquisition package and external review.

## Minimum acquisition package

For each campaign, retain immutable copies of:

1. Native instrument files for every sample, angle and polarization; calibration-standard certificates and calibration runs; background/no-sample and reference-target runs; repeat scans; and phase-reference files if coherent-field claims are intended.
2. Calibrated per-sample observations and intermediate outputs needed to reproduce calibration, background removal, time-gating, angular correction and integration.
3. Campaign protocol, geometry/survey, specimen records, environmental conditions, instrument configuration, processing code/configuration and operator/lab record.
4. Frozen model/material mapping, expected strata, complete train/validation/test split manifest, its hash, external holdout-verification record, uncertainty method and prediction rows.
5. Rights and permissions for raw data, calibrated data, figures and derived artifacts, including rights holder, allowed research/publication/redistribution, attribution and restrictions.

Inventory every source, calibrated result, calibration input and processing input with stable relative path, byte size, media type, SHA-256, role, campaign ID and links to dependent sample/specimen, unit and calibration records. Preserve original files; never substitute processed exports for source files. The material-scoring evaluator does not check this inventory; the separate proposal-only v0.2/v0.3 package verifier checks inventoried bytes, hashes, paths and record joins, but does not authenticate the source or establish scientific validity.

## Required record fields

### Campaign, instrument and calibration

Record campaign ID; UTC acquisition start/end; lab/site; operator role or pseudonymous ID; instrument make/model/controlled serial and firmware; Tx/Rx chain; antenna IDs, patterns, gain and polarization basis; cable/fixture IDs; carrier and sweep/span settings; waveform, power, IF bandwidth, averaging and receiver settings; chamber/site geometry; environmental conditions; coordinate frame; survey method and uncertainty; and instrument traceability certificates.

For each calibration/reference run, record calibration ID and file IDs, method, standards/certificates, validity interval, reference plane, de-embedding, drift checks, background subtraction, time-gating/window and limits, processing version/configuration hash, and derivation of calibrated power and angular response. Code requires a nonempty `measurement_calibration_id`; it does not validate these supporting records or calibration validity.

### Specimen and physical independent unit

For each specimen, record specimen ID; material claim/source and composition/grade; layer stack, thickness, backing and dimensions; preparation/finish; roughness statistic, instrument, scale/cutoff and uncertainty; moisture, temperature and conditioning; controlled photo IDs; custody/preparation record; and exact mapping to any named scene material.

Before collection, define the physical independent acquisition unit in the external protocol. Give each unit a `measurement_unit_id` and a separate `measurement_unit_provenance_id` linking to a signed/controlled record. State which rows are repeated scans, angles, repositionings, or independent specimens/acquisitions and which must remain clustered. Do not treat repeated angles, sweeps, files, bins or re-exports as independent because their row IDs differ. An external reviewer must attest that the units are physically independent and suitable for the intended inference. The evaluator checks ID presence and consistency only; it cannot verify this claim.

### Geometry, strata and observations

For every sample/row, record sample ID; incidence group ID; unit and provenance IDs; material ID; polarization (`V`/`H`); incidence-stratum ID and angle; receive/scatter angle or angular-bin definition; Tx/Rx coordinates and orientations in a declared frame; antenna boresight and polarization alignment; path-length/reference geometry; frequency; acquisition time; split; and links to raw, calibration, background and calibrated files.

Preregister the complete **material × polarization × incidence-stratum** set. Define each stratum's angle range and discrete incidence angles before inspecting held-out outcomes. Retain the full intended sweep, including below-detection, invalid, saturated and missing observations with explicit status/reason. The evaluator enforces finite values, consistent bin edges, complete rows for declared held-out groups and exact expected-strata coverage; it cannot establish that the chosen strata cover the intended use domain.

### Observable and uncertainty

For integrated reflected-plus-diffuse power, declare units and reference plane. Preserve calibrated linear power and per-angle-bin linear powers before dB conversion; state whether direct/specular energy is included and how it is separated or retained. Define angular coordinate, bin edges and semantics. Evaluator input v2 requires integrated linear power per bin on a complete zenith/azimuth sphere, one declared coordinate frame, and solid-angle values matching each cell's spherical geometry. The score is the supremum difference between normalized 2D cumulative-power grids; missing bins are unresolved and cannot be imputed as zero. This is a candidate project-defined replacement statistic, not yet approved for qualification. Freeze its coordinate conventions and measurement corrections through independent review before use.

Per row, retain measured and predicted integrated power, measured and predicted angular-power arrays, and the nominal 95% prediction interval `[lower_db, upper_db]`. Document measurement uncertainty separately from model/predictive uncertainty: calibration/drift, receiver noise/dynamic range, background removal, repeatability, alignment/angle survey, antenna response, integration/binning, specimen variability, processing/model uncertainty, dependence assumptions and propagation method. Preserve repeat-level data for independent recomputation. Code requires only a nonempty `uncertainty_description`, checks interval ordering and computes coverage/width; it does not validate the uncertainty budget or interval calibration.

Record the fitted material-model parameters used to produce each prediction, including Sionna's scattering coefficient `S`, angular-pattern parameters, fitting objective/method, optimizer/configuration, initialization and parameter uncertainty or covariance. Bind these values to material/specimen, model version and prediction rows. A label such as “fitted diffuse material” is insufficient to reproduce or audit the prediction.

For any 3D scattering or angular-pattern claim, acquire and retain bistatic azimuth **and** zenith coverage with declared Tx/Rx angle conventions, coordinate frame, polarization basis, angular sampling and missing/invalid-bin status. State whether arrays represent power per angular bin or density per solid angle, and document the solid-angle/Jacobian correction used when converting between them. A one-dimensional azimuth/elevation sweep or normalized 1D CDF cannot establish a 3D scattering-pattern result. The v2 scorer enforces the full-sphere grid and cell solid angles but does not establish that the laboratory coordinate frame, calibration, or collected pattern is scientifically adequate.

## Holdout binding

Before collecting or unblinding qualification measurements, freeze and independently timestamp/sign the protocol, physical-unit definition, model ID/hash, calibration ID, surface description, material mapping, target frequency, expected strata, angular conventions, uncertainty method, all sample/unit/group/split assignments, bootstrap seed and replicate count.

The canonical `full_split_manifest` must contain every train/validation/test group and its complete globally unique sample IDs. Each test-sample assignment binds sample, group, material, polarization, incidence stratum/angle, unit and provenance IDs. Include `full_split_manifest_sha256` over canonical JSON and a `split_manifest_external_verification_id` resolving to an independent custodian's signed/timestamped record. The scoring `incidence_groups` must list all complete held-out groups and sample IDs. Inventory the manifest, external record and hashes. A hash proves consistency of supplied bytes, not authorship, pre-registration, completeness or absence of test access; those require external verification.

## Frozen evaluator interface and criteria

The legacy schema is `loslab.g3_material_calibration_predictions.v1`; it retains the original 1D proposal contract. New v0.3 acquisition proposals bind `loslab.g3_material_calibration_predictions.v2`, which retains the same top-level provenance/split fields and replaces 1D angular arrays with full-sphere integrated-power matrices, per-cell solid angles, coordinate-frame identity and explicit missing-bin status. Exact fields are in `datasets/schema/g3-material-calibration-predictions.v2.schema.json` and semantic checks are in `evaluation/g3_material_calibration.py`.

The Draft 2020-12 schemas for v1 and v2 are exercised against synthetic fixtures in `tests/test_evaluation/test_g3_material_calibration.py`. They are optional structural preflights, not invoked by the evaluator CLI, and reject unknown fields. V2's schema checks local structure and edge ranges; the evaluator remains authoritative for full-sphere edge ordering, matrix dimensions, finite nonnegative powers, exact solid-angle geometry, missing-bin rejection, canonical manifest hashes, cross-record joins/completeness and numeric margins. Schema or evaluator consistency does not authenticate measurements or qualify a material model.

The frozen G3.03 numeric margins remain per material/polarization/incidence stratum on complete held-out incidence groups: median absolute integrated reflected-plus-diffuse power error **≤1.5 dB**; 95th percentile absolute error **≤3 dB**; angular-pattern distance **≤0.10**; nominal 95% prediction-interval coverage **90–98%**; median interval width **≤3 dB**. The earlier angular statistic and the v2 2D joint-CDF candidate are not interchangeable; independent scientific review must approve the statistic before qualification. The shared multiplicity contract in `evaluation/g3_joint_acceptance.py` now supports a v3 roster covering material, complex-CSI, propagation, and four paired receiver-impact intervals per expected LOS/NLOS stratum under one 95% family. Both material scorer input and the v0.3 proposal verifier accept the v3 family while checking its canonical digest and material roster; a synthetic package test confirms the verifier retains proposal-only status. The v4 assumption-only planner runs nested receiver cluster/block bootstrap assurance by passing each simulated study through the receiver evaluator itself. It reports separate material-unit and scene-cluster designs and leaves the complete G3 sample size null. All scenario probabilities and per-block outcomes remain hypothetical; hash consistency does not establish preregistration, physical independence, representativeness, or external authorship. Give independent units equal weight, cluster-resample their rows, and require every adjusted interval to lie within its corresponding margin. Fewer than 20 material units is not assessable; 20 remains a pilot floor, not a powered sample size. External verification and G3 qualification remain prerequisites.

Complex-field/CSI claims require phase-referenced held-out measurements and the already-frozen separate complex-CSI criteria; power/angular agreement alone is insufficient. For reproducibility, bind each complex-CSI observation to exact evaluated NR subcarrier indices and frequency offsets, antenna ports, OFDM symbols, absolute delay reference, carrier/LO phase reference and phase convention, and the calibration/de-embedding state. Record the noise-derived `H_floor` inputs and method used by the frozen CSI scoring contract, with the resulting value per applicable receiver/configuration; do not infer or replace that floor from the held-out prediction error. Preserve complex-valued measured and predicted CSI before any magnitude, phase-unwrapping, or normalization transform. The current G3.03 material evaluator does not ingest or validate this CSI package. Frequency transfer must be supported by target-frequency evidence under the same existing limits.

## Lab checklist

- [ ] Protocol frozen before acquisition: 3.5 GHz, named materials/specimens, polarizations, incidence strata and angular coverage.
- [ ] Specimen and physical-unit provenance is traceable; repeated measurements are clustered; independence is externally attested.
- [ ] Raw, reference, calibration, repeat and calibrated files are immutable, inventoried, hashed and linked to samples/calibration dependencies.
- [ ] Calibration traceability, reference plane, antenna/polarization response, geometry, environment and processing are reproducible.
- [ ] Power/angular fields have units, coordinate/bin conventions, detection/missingness status and retained linear-domain source values.
- [ ] Measurement and prediction uncertainty are separately documented and reproducible.
- [ ] Worst-stratum probabilities and IID bootstrap scenarios have an independent basis and are preregistered; the hash-bound planning report is retained; the complete multi-estimand design is reconciled before acquisition.
- [ ] All split assignments are frozen, hashed and externally verified before holdout access; expected strata are complete.
- [ ] Material-to-scene mapping is specimen-specific; transfer assumptions and support limits are explicit.
- [ ] Raw/calibrated/derived-data rights, publication and redistribution permissions, third-party rights and attribution are recorded.
- [ ] OSM-derived scene inputs/outputs receive a separate review under the existing ODbL policy; lab-data permission does not clear scene-derived release rights.
- [ ] Independent scientific reviewer signs calibration, physical-unit, holdout, uncertainty and intended-claim attestations.
- [ ] G3.03 and relevant release gates remain closed until all frozen evidence and release conditions are satisfied.

## Enforcement boundary

| Item | Enforced by current code | External evidence needed |
|---|---|---|
| 3.5-GHz value, required IDs/descriptions, hash syntax, bootstrap seed/count | Presence, syntax, target tolerance | Truth of IDs, model provenance and calibration validity |
| Expected strata and test sample/group bindings | Set equality, uniqueness, row completeness, value consistency, canonical manifest hash | Actual completeness, preregistration and representativeness |
| Unit/provenance IDs | Presence/consistency; equal-unit summaries and cluster bootstrap | Physical independence, specimen authenticity and correct clustering |
| Prediction/measurement arrays and intervals | Finite/range/shape checks, interval ordering, metric computation | Physical calibration, uncertainty validity, antenna and power corrections |
| Fitted material parameters and 3D scattering coverage | Not ingested by current evaluator | Fitted `S` and angular parameters, fit provenance/uncertainty, bistatic azimuth/elevation support and solid-angle convention |
| Complex CSI evidence | Not ingested by current evaluator | Exact NR resource indices, antenna ports/symbols, absolute delay and phase reference, noise-derived `H_floor`, phase-referenced held-out measurements |
| Split verification ID | Nonempty ID | Custodian record, signature/time and no holdout leakage |
| Raw files, calibration chain, specimen/geometry, rights | Not ingested by G3 evaluator | Lab records, scientific and rights review |
| Public release | Separate packager checks declared hashes and required ODbL metadata/classification fields | Correct legal classification and rights conclusions |

## Exact repository locations

This draft's proposed location is `docs/research/G3_03_LAB_ACQUISITION_DATA_CONTRACT_DRAFT.md`.

### Implemented proposal-only acquisition-package interfaces (v0.2/v0.3; not qualification)

The versioned v0.2 and v0.3 schemas and semantic verifier are available, but packages remain explicitly `proposal_only`; verifier success returns `qualification_status: NOT_QUALIFIED`. Version 0.2 preserves the earlier structural format. Version 0.3 adds full-sphere angular geometry checks and binds the candidate evaluator-input v2. The 2D angular distance is a proposed project-defined scoring method, not an approved change to frozen G3 acceptance criteria; it requires scientific review and preregistration before qualification use.

- Schemas: `datasets/schema/g3.03-acquisition-package.v0.2.schema.json`, `datasets/schema/g3.03-acquisition-package.v0.3.schema.json`, and `datasets/schema/g3.03-acquisition-package.v0.4.schema.json` (JSON Schema Draft 2020-12). Synthetic contract tests are in `tests/test_datasets/test_g3_acquisition_package_v0_2.py` and `tests/test_evaluation/test_g3_acquisition_package_verifier.py`; they are software fixtures, not measurement evidence.
- The contract links campaign, instrument, protocol, specimen, acquisition-unit, calibration, sample, fit, angular/CSI measurement, uncertainty, rights, and external-attestation records to an inventoried set of package-relative files.
- `evaluation/g3_acquisition_package.py` checks bounded package/schema reads; inventory paths and exact bytes; declared SHA-256/size; unique identifiers and record references; per-file rights coverage; and bindings for the scorer input, split manifest, and evaluator output. It checks that the scorer input embeds the canonical split manifest and that the output echoes the exact scorer-input hash, then validates available strata and row-crosswalk joins.
- `evaluation/g3_signed_attestation.py` verifies a detached OpenSSH signature over a canonical envelope bound to the exact package bytes and the package-inventoried hashes/sizes of the attested subject files. It requires both a trusted `allowed_signers` file and a canonical signer-role policy supplied outside the package, and uses namespace `loslab-g3-acquisition-attestation-v1@github.com`; it also runs the package consistency verifier first. The sidecar does not change a proposal record's `planned_not_signed` status or qualify evidence. The policy authorizes a principal for specific organizations, attestation types, and roles; its governance and the truth of signed claims remain external.
- `scripts/audit_g3_attestation_set.py` checks exactly one verified sidecar per package-declared external-attestation record, reusing one package/trust snapshot for the set. It rejects missing or duplicate coverage and reports signer count without asserting independence. The report remains `NOT_QUALIFIED`.
- The verifier reports `consistent_proposal_only`, never scientific admission. It does not rerun scoring or prove measurement authenticity, calibration validity, physical independence, preregistration time, holdout secrecy, legal-rights sufficiency, signature authenticity, or representativeness. Those require independent lab, custodian, scientific, and rights review.
- The contract carries fields for the campaign/instrument configuration, specimen and acquisition-unit provenance, fitted parameters, 3D angular observations, complex CSI, uncertainty, split/scoring artifacts, and rights. Their presence and cross-references do not establish that the protocol or data satisfy the scientific meaning required by the frozen criteria.

The v0.2 schema and verifier close earlier wire-format omissions for campaign/instrument records, the split manifest, scorer rows, and evaluator-output binding; v0.3 additionally validates angular full-sphere coverage and solid-angle cells, requires the shared material/complex-CSI acceptance-family declaration, and requires each CSI noise-input inventory entry to use the `noise_measurement` role. Versions v0.1–v0.3 retain the historical declared-only `h_floor` semantics. Version v0.4 defines `h_floor.amplitude_linear` as `sqrt(mean(abs(noise_only_cfr_estimates)**2))` over hash-verified, same-sample complex NPY 1.0/2.0 artifacts whose array shape matches the CSI row. It requires the declared value to match the recomputation and rejects non-finite values, non-complex arrays, truncated/trailing payloads, shape mismatch, and mismatched sample binding. This establishes arithmetic, byte integrity, and declared shape consistency only; it cannot establish identical estimator behavior, calibration, or scientific representativeness. The separate signed-attestation verifier can authenticate a detached reviewer statement against exact package and subject-file hashes, but does not establish its truth or replace the planned package fields with signed status. No calibrated campaign, verified independent units, pre-acquisition holdout attestation, externally validated uncertainty budget, institutional signer trust, or rights determination is present. Do not treat a synthetic fixture, valid signature, or consistent report as qualification data.

#### Detached external-attestation signature envelope v1

The verifier accepts a canonical UTF-8 JSON sidecar with schema `loslab.g3_signed_attestation.v1`, signed as raw bytes using OpenSSH SSHSIG namespace `loslab-g3-acquisition-attestation-v1@github.com`. The envelope contains the package's exact SHA-256, an existing `external_attestations` record ID and matching type/claim/role, sorted subject-file IDs with the inventory SHA-256 and byte size, signer principal, signing organization, and UTC signing time. Keep the package immutable after creating the envelope. The detached `.sig`, trusted OpenSSH `allowed_signers`, and signer-role policy are separate inputs; never take trust inputs from the package being reviewed.

The canonical trust-policy file has schema `loslab.g3_attestation_trust_policy.v1`, a `policy_id`, and a unique `signers` array. Each entry binds one `principal` to an exact `organization`, `allowed_attestation_types` array, and `allowed_signer_roles` array. The verifier checks the signed envelope against this mapping and records the policy-file SHA-256. The policy file must be obtained and governed through the institution's independent trust process; its presence or hash does not prove the institution's authority or the signer's independence.

    ssh-keygen -Y sign -f reviewer_ed25519 -n loslab-g3-acquisition-attestation-v1@github.com attestation.json
    uv run --no-sync --locked python -m evaluation.g3_signed_attestation package.json attestation.json attestation.json.sig --allowed-signers /trusted/review/allowed_signers --signer-policy /trusted/review/g3-attestation-trust.json
    uv run --no-sync --locked python scripts/audit_g3_attestation_set.py package.json attestations/*.json --allowed-signers /trusted/review/allowed_signers --signer-policy /trusted/review/g3-attestation-trust.json

The verifier returns `qualification_status: NOT_QUALIFIED` even when the signature is valid. Its report does not independently prove physical independence, preregistration timing, calibration truth, legal sufficiency, or signer independence. The batch command requires one sidecar for every declared external-attestation record. The current v0.2/v0.3 package schemas still declare embedded attestations as planned and unsigned; a future wire-format revision is needed before signed status becomes part of the package itself. Fourteen tests cover both package versions, complete/incomplete sidecar sets, signature and digest tampering, untrusted principals, inconsistent inventory, hard links, and trust-policy role/type/organization restrictions.

For later implementation/review:

- `evaluation/g3_material_calibration.py` — versioned material-scoring evaluator; v2 is a candidate angular scoring contract and is not yet approved for qualification.
- `evaluation/g3_signed_attestation.py`, `scripts/audit_g3_attestation_set.py`, and `tests/test_evaluation/test_g3_signed_attestation.py` — single and package-set detached-signature verification; signature completeness is not qualification.
- `tests/test_evaluation/test_g3_material_calibration.py` — evaluator schema, consistency and fail-closed tests.
- `tests/test_datasets/test_g3_acquisition_package_schema.py` and `tests/test_datasets/test_g3_acquisition_package_v0_2.py` — proposal-schema, verifier, and synthetic contract checks.
- `datasets/schema/g3.03-acquisition-package.v0.2.schema.json`, `datasets/schema/g3.03-acquisition-package.v0.3.schema.json`, `datasets/schema/g3.03-acquisition-package.v0.4.schema.json`, and `evaluation/g3_acquisition_package.py` — proposal-only wire formats and consistency verifier.
- `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` — frozen G3.03 criteria, independence, holdout, bootstrap and release gates; this draft does not revise them.
- `docs/research/G3_SCENE_FAMILY_LICENSE_AUDIT_2026-10-23.md` — literature/material evidence and current G3.03 disposition.
- `docs/research/licenses/ODBL_OSM_SCENE_RELEASE_POLICY_2026-10-23.md`, `docs/research/licenses/odbl-osm-scene-release-policy-20261023.json`, `datasets/public_release.py`, and `tests/test_datasets/test_public_release.py` — separate scene-lineage/public-release rights gate.
- `docs/research/MVP_STATUS.md` — update only when evidence changes; this checklist alone is not gate progress.

Both package versions are reviewable scaffolding only. Before either is accepted for a real campaign, the lab and independent reviewers must settle protocol-specific conventions and trust policy, supply actual source/calibration/acquisition evidence, verify the external holdout and independence attestations, and complete the preregistered multi-estimand design. The candidate v2 angular scoring method must be reviewed and explicitly version-approved before it can be used as an acceptance criterion.
### Reproducible joint acquisition-design planning (v2; not qualification)

The `scripts/plan_g3_acquisition_design.py` command retains v2/v3 input compatibility and adds the v4 contract in `datasets/schema/g3.03-acquisition-design-assumptions.v4.schema.json`. V4 requires the shared family-v3 roster and accepts finite-support hypothetical cluster outcomes containing paired candidate/challenger CRC results across the frozen SNR grid. Every simulated study uses the production receiver evaluator's nested cluster/block bootstrap and all four receiver intervals. Exact binomial lower bounds quantify Monte Carlo assurance, with the joint pass event evaluated across declared receiver strata. The report keeps material acquisition units distinct from scene clusters, marks every design as planning-only, and never emits a complete combined G3 sample size. Example v2/v3 use:

    uv run --no-sync --locked python scripts/plan_g3_acquisition_design.py datasets/schema/examples/g3.03-acquisition-design-assumptions.synthetic-example.json --output g3-acquisition-design-report.json

The schema fixes family-wise alpha at 0.05 and requires bootstrap candidate sizes of at least 20 independent material units, preserving the frozen 95% criterion and evaluator floor. The report binds the exact configuration, family, schema, planner and runner. It returns NOT_QUALIFIED and reports assumption-driven counts without collapsing distinct material-unit and scene-cluster populations into one sample size. The checked-in example is explicitly synthetic: under its hypothetical assumptions, the conservative design gives 12,475 material units per stratum and 6,147 CSI clusters per stratum, while its simple two-point material bootstrap scenario selects 20 among the two evaluated candidates. This disparity exposes assumption sensitivity; neither count is a lab recommendation, complete joint assurance, or power claim. Replace every hypothetical assumption with an independently justified, preregistered design input before campaign planning.

Independent justification and preregistration of the worst-stratum probabilities and IID bootstrap scenarios, together with reconciliation of the complete multi-estimand design, remain prerequisites before qualification.

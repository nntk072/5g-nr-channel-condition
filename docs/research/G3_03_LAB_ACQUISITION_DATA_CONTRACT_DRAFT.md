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

Inventory every source, calibrated result, calibration input and processing input with stable relative path, byte size, media type, SHA-256, role, campaign ID and links to dependent sample/specimen, unit and calibration records. Preserve original files; never substitute processed exports for source files. The evaluator currently checks no file inventory or file hash.

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

For integrated reflected-plus-diffuse power, declare units and reference plane. Preserve calibrated linear power and per-angle-bin linear powers before dB conversion; state whether direct/specular energy is included and how it is separated or retained. Define angular coordinate, bin edges and semantics. The evaluator normalizes each supplied nonnegative angular array by its sum, cumulatively sums it, and computes CDF KS distance. It does not validate physical interpretation, solid-angle/Jacobian correction or consistency of measurement/prediction corrections.

Per row, retain measured and predicted integrated power, measured and predicted angular-power arrays, and the nominal 95% prediction interval `[lower_db, upper_db]`. Document measurement uncertainty separately from model/predictive uncertainty: calibration/drift, receiver noise/dynamic range, background removal, repeatability, alignment/angle survey, antenna response, integration/binning, specimen variability, processing/model uncertainty, dependence assumptions and propagation method. Preserve repeat-level data for independent recomputation. Code requires only a nonempty `uncertainty_description`, checks interval ordering and computes coverage/width; it does not validate the uncertainty budget or interval calibration.

Record the fitted material-model parameters used to produce each prediction, including Sionna's scattering coefficient `S`, angular-pattern parameters, fitting objective/method, optimizer/configuration, initialization and parameter uncertainty or covariance. Bind these values to material/specimen, model version and prediction rows. A label such as “fitted diffuse material” is insufficient to reproduce or audit the prediction.

For any 3D scattering or angular-pattern claim, acquire and retain bistatic azimuth **and** elevation coverage with declared Tx/Rx angle conventions, coordinate frame, polarization basis, angular sampling and missing/invalid-bin status. State whether arrays represent power per angular bin or density per solid angle, and document the solid-angle/Jacobian correction used when converting between them. A one-dimensional azimuth/elevation sweep or a normalized 1D CDF cannot establish a 3D scattering-pattern result. The present evaluator accepts one-dimensional angular arrays and does not enforce 3D coverage or solid-angle weighting.

## Holdout binding

Before collecting or unblinding qualification measurements, freeze and independently timestamp/sign the protocol, physical-unit definition, model ID/hash, calibration ID, surface description, material mapping, target frequency, expected strata, angular conventions, uncertainty method, all sample/unit/group/split assignments, bootstrap seed and replicate count.

The canonical `full_split_manifest` must contain every train/validation/test group and its complete globally unique sample IDs. Each test-sample assignment binds sample, group, material, polarization, incidence stratum/angle, unit and provenance IDs. Include `full_split_manifest_sha256` over canonical JSON and a `split_manifest_external_verification_id` resolving to an independent custodian's signed/timestamped record. The scoring `incidence_groups` must list all complete held-out groups and sample IDs. Inventory the manifest, external record and hashes. A hash proves consistency of supplied bytes, not authorship, pre-registration, completeness or absence of test access; those require external verification.

## Frozen evaluator interface and criteria

The existing schema is `loslab.g3_material_calibration_predictions.v1`. Enforced top-level fields include: `target_frequency_hz` (3.5e9 within 1 Hz), `model_id`, `model_sha256`, `measurement_calibration_id`, `material_mapping_id`, `surface_description`, `uncertainty_description`, `measurement_unit_independence_description`, `cluster_bootstrap_seed`, `cluster_bootstrap_replicates` (≥1000), `expected_strata`, `full_split_manifest`, canonical manifest SHA-256, `split_manifest_external_verification_id`, `incidence_groups` and `test_predictions`. Exact nested fields are in `evaluation/g3_material_calibration.py`.

The Draft 2020-12 structural schema for this existing payload is `datasets/schema/g3-material-calibration-predictions.v1.schema.json`, exercised against the synthetic `_payload()` in `tests/test_evaluation/test_g3_material_calibration.py`. It is an optional stricter preflight and is not invoked by the evaluator CLI; in particular, it rejects unknown fields even though the evaluator ignores them. It checks JSON structure, required fields, local types/shapes and digest syntax only. The evaluator remains authoritative for semantic numeric validity (including nonblank identifiers and V/H polarization), canonical manifest hash equality, cross-record joins/uniqueness/completeness, and the frozen numeric criteria; schema validity alone does not establish those checks or qualify measurements.

Keep the existing G3.03 criteria unchanged, per admitted material/polarization/incidence stratum on complete held-out incidence groups: median absolute integrated reflected-plus-diffuse power error **≤1.5 dB**; 95th percentile absolute error **≤3 dB**; normalized angular-power CDF KS **≤0.10**; nominal 95% prediction-interval coverage **90–98%**; median interval width **≤3 dB**. Give independent units equal weight, cluster-resample their rows, and require the specified family-wise-adjusted two-sided 95% percentile-bootstrap intervals to lie within corresponding margins. Fewer than 20 units is not assessable; 20 is a pilot floor, not a powered sample size. The complete multi-estimand power/precision design and external verification remain prerequisites. Current evaluator output remains pilot-only/unverified for admission.

Complex-field/CSI claims require phase-referenced held-out measurements and the already-frozen separate complex-CSI criteria; power/angular agreement alone is insufficient. For reproducibility, bind each complex-CSI observation to exact evaluated NR subcarrier indices and frequency offsets, antenna ports, OFDM symbols, absolute delay reference, carrier/LO phase reference and phase convention, and the calibration/de-embedding state. Record the noise-derived `H_floor` inputs and method used by the frozen CSI scoring contract, with the resulting value per applicable receiver/configuration; do not infer or replace that floor from the held-out prediction error. Preserve complex-valued measured and predicted CSI before any magnitude, phase-unwrapping, or normalization transform. The current G3.03 material evaluator does not ingest or validate this CSI package. Frequency transfer must be supported by target-frequency evidence under the same existing limits.

## Lab checklist

- [ ] Protocol frozen before acquisition: 3.5 GHz, named materials/specimens, polarizations, incidence strata and angular coverage.
- [ ] Specimen and physical-unit provenance is traceable; repeated measurements are clustered; independence is externally attested.
- [ ] Raw, reference, calibration, repeat and calibrated files are immutable, inventoried, hashed and linked to samples/calibration dependencies.
- [ ] Calibration traceability, reference plane, antenna/polarization response, geometry, environment and processing are reproducible.
- [ ] Power/angular fields have units, coordinate/bin conventions, detection/missingness status and retained linear-domain source values.
- [ ] Measurement and prediction uncertainty are separately documented and reproducible.
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

### Proposed acquisition-package interface (v0.1; not frozen or accepted)

This is an explicit wire-format proposal for review. It does not change the evaluator, frozen acceptance criteria, admission state, or authorization for measurements or RT generation. Do not describe this proposal as an agreed acquisition contract.

- Schema: `datasets/schema/g3.03-acquisition-package.v0.1.schema.json` (JSON Schema Draft 2020-12); fictional illustrative fixture: `datasets/schema/examples/g3.03-acquisition-package.v0.1.example.json`.
- IDs are stable, case-sensitive ASCII tokens; file IDs use the `file.` prefix. References point to top-level records. Paths are package-relative POSIX paths without `..` segments. Hashes are lowercase SHA-256.
- Quantities use explicit SI units: frequency Hz, length m, UTC ISO-8601 time, power W, angles degrees, solid angle sr, phase radians, and dimensionless complex channel gain. The 3D grid uses zenith theta `[0,180]` degrees and azimuth phi `[-180,180)` degrees, with arrays ordered `[theta_bin][phi_bin]`; power is integrated per cell, and solid-angle measure is recorded for each cell.
- CSI values are flattened `[real, imaginary]` pairs in C row-major order over `rx_port, tx_port, subcarrier, ofdm_symbol`. Records bind exact resource indices and frequency offsets, symbol indices, absolute delay reference, carrier phase reference/convention, and linear-amplitude `H_floor` method and noise inputs.
- File, specimen, acquisition-unit, calibration, fit, angular-observation, CSI, rights and attestation records are separate and linked by stable IDs. Fit records carry named parameter estimates, units and standard uncertainties; optional covariance is preserved by file reference.
- External signatures, physical independence, calibration truth/validity, ownership and rights sufficiency remain attestations. JSON Schema can check shape, syntax and local consistency but cannot prove those external facts or validate the attestations.

The schema and fixture do not add an evaluator loader or validate measurement results. Before freezing any interface, reviewers still need to decide whether the proposed angular and phase conventions suit all intended labs, resolve detailed calibration/covariance fields and signature trust policy, define rights/status vocabularies, and reconcile CSI fields with the separate frozen scoring contract. The fixture contains fictional values only and is not measurement evidence.

**Known v0.1 scope gap:** This proposal is not yet a complete G3 qualification package. It does not serialize full campaign/instrument configuration records, the complete preregistered split and expected-strata manifest, scored prediction/interval rows, or a binding to an evaluator output. Those items remain required by the acquisition checklist and frozen evaluator contract; do not use v0.1 alone to claim package completeness or admission readiness. A successor must add or explicitly bind these structures before an acquisition-package loader is implemented.

For later implementation/review:

- `evaluation/g3_material_calibration.py` — existing frozen evaluator; unchanged by this proposal.
- `tests/test_evaluation/test_g3_material_calibration.py` — evaluator schema, consistency and fail-closed tests.
- `tests/test_datasets/test_g3_acquisition_package_schema.py` — proposal-schema and fixture checks.
- `datasets/schema/g3.03-acquisition-package.v0.1.schema.json` — versioned proposal-only JSON Schema.
- `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` — frozen G3.03 criteria, independence, holdout, bootstrap and release gates; this draft does not revise them.
- `docs/research/G3_SCENE_FAMILY_LICENSE_AUDIT_2026-10-23.md` — literature/material evidence and current G3.03 disposition.
- `docs/research/licenses/ODBL_OSM_SCENE_RELEASE_POLICY_2026-10-23.md`, `docs/research/licenses/odbl-osm-scene-release-policy-20261023.json`, `datasets/public_release.py`, and `tests/test_datasets/test_public_release.py` — separate scene-lineage/public-release rights gate.
- `docs/research/MVP_STATUS.md` — update only when evidence changes; this checklist alone is not gate progress.

The v0.1 proposal is reviewable scaffolding only. Any accepted successor must cover raw/calibrated file inventory, sample-file dependencies, specimen/acquisition provenance, calibration chain, geometry/uncertainty, fitted material parameters, 3D angular coverage, the separate complex-CSI evidence package and rights, while preserving frozen evaluator behavior unless a separately reviewed versioned change is approved.

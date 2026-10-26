# Command wrappers

Reusable research logic belongs in importable packages. This directory is for
thin data, training, evaluation, scene, and research command wrappers when they
are needed for batch or scheduler entry points. The public local CLI is
currently `python -m loslab` (or the installed `loslab` command).

## Frozen UMCD corpus audit

`audit_umcd_corpus.py` accepts frozen corpus inventory v2/v3/v4. V4 requires an
explicit `scene_inventory` mapping scene IDs to exact parent episode IDs and
reports the whole-scene split check alongside split lineage. The result is
input-scoped; the inventory still needs independent completeness and authority
review before any release claim. Signed v4 inventories use the dedicated
`umcd-corpus-inventory-v4@github.com` namespace; v3 keeps its v3 namespace.

## G4 normative-reference byte audit

`audit_g4_standard_references.py` checks that locally downloaded ETSI PDF files
match the external source identities in
`docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json`:

```bash
python scripts/audit_g4_standard_references.py \
  docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json \
  /path/to/privately-downloaded-etsi-references
```

The manifest records source URLs, versions, byte counts, and SHA-256 digests;
the PDFs are not stored in or redistributed by this repository. The audit
rejects unsafe or changing file inputs and reports `NOT_QUALIFIED` even when
all bytes match. It verifies source identity only; it does not extract or
compare normative parameter values or establish channel-model conformance.
The manifest and emitted report contracts are defined in
`datasets/schema/g4-standard-reference-manifest.v1.schema.json` and
`datasets/schema/g4-standard-reference-audit.v1.schema.json`.

## G4 scenario GBSM generation

`loslab scenario-gbsm-data` generates explicitly configured UMi, UMa, RMa,
InH, or InF standardized-model drops. It writes both a content-addressed
scenario run store and one UMCD `EpisodeBundle` per named drop. Drops generated
in the same run share a split group; the command makes no claim about site-level
independence across runs. Scenario LOS is retained as a standardized-model
state, while geometric LOS and the physical RT path oracle remain absent.
The v2 report lists the run-store and episode-store manifests and remains
`SYNTHETIC_STANDARDIZED_MODEL_DATA_NOT_QUALIFIED`. See
`docs/research/G4_SCENARIO_GBSM_GENERATION_CLI_V1.md` for the bounded config
contract and limitations.

## RT trajectory forecast pilot split provenance

`evaluate_rt_nr_trajectory_forecasts.py` can evaluate a fixed baseline against
either the legacy unsigned diagnostic split lock or a signed UMCD split-policy
registry. For signed use, supply the detached signature, an independently
trusted OpenSSH `allowed_signers` file, and the expected signer principal:

```bash
python scripts/evaluate_rt_nr_trajectory_forecasts.py \
  /path/to/nr-trajectory-store \
  /path/to/split-registry.json \
  /path/to/forecast-report.json \
  /path/to/trajectory-spec.json \
  --split-signature /path/to/split-registry.json.sig \
  --trusted-split-allowed-signers /path/to/allowed_signers \
  --split-signer-id research@example.org
```

The signed registry must contain child bindings of kind
`nr_trajectory_sample`; the supplied trajectory specs must cover that exact
child roster. The report records exact registry, signature, and trust-file
digests. Omitting all three signature options uses the legacy path and marks
its split provenance diagnostic-only. Neither mode makes this small fixed
baseline a forecasting qualification result.
# Reproducible research runs

Use `run_rt_cross_scene_pilot.sh` to generate and evaluate the small, exploratory
three-scene RT pilot. Run it from a checkout with the pinned Sionna RT 2.2.0
environment active and Mitsuba's LLVM variant available:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  bash scripts/run_rt_cross_scene_pilot.sh outputs/rt-cross-scene-pilot-rerun
```

The script refuses to overwrite an existing output directory. Each NPZ row
contains its scene, OFDM-grid, and solver configuration. CPU RT can differ by
small floating-point amounts between runs, so reproduce labels, groups, and
metrics within numerical tolerance rather than expecting identical file hashes.
The three built-in scenes and 40 links per scene are not a confirmatory or
standards-conformance dataset.

Use `run_tdl_pusch_bler_curve.py` for a seeded Sionna TDL-to-PUSCH BLER curve
on the CUDA PHY environment:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  python scripts/run_tdl_pusch_bler_curve.py \
  outputs/tdl-pusch-bler-run.json \
  --frames 1024 --batch-size 128 --snr-db 0 6 12 18 24 --seed 31000
```

The script refuses to overwrite a result, records codeword counts and Wilson
intervals, and checks that transmitted bits and TDL channel truth match across
SNR points. Its current `OFDMChannel` path models TDL fading with perfect
synchronization and no sample-domain ISI/ICI; it is not a time-domain waveform
receiver-capability or standards-conformance claim.

Use `run_tdl_pusch_time_bler_curve.py` to evaluate sample-domain TDL fading and
the PUSCH waveform receiver, including the sampled channel filter memory:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  python scripts/run_tdl_pusch_time_bler_curve.py \
  outputs/tdl-pusch-time-bler-run.json \
  --frames 1024 --batch-size 64 --snr-db 0 6 12 18 24 --seed 41000 \
  --n-size-bwp 12 --subcarrier-spacing-khz 30
```

The default 12-PRB grid provides a 13-sample shortest cyclic prefix for the
configured 13-sample discrete channel memory. The runner refuses a configuration
whose discrete channel memory exceeds its shortest CP, reuses the same waveform,
TDL realization, and standardized noise seed at each SNR, and records measured
transmit waveform power, per-point Wilson diagnostics, batch-seed cluster-bootstrap
intervals, and hashes. The cluster interval resamples whole generated batches and
is null when fewer than two batches are available; it remains conditional on this
synthetic link and does not estimate independent-scene uncertainty. The curve is a
simulation result with ideal synchronization; it does not validate blind timing,
fractional timing/CFO recovery, external propagation measurements, or standards
conformance.

For a controlled mobility comparison, generate control and treatment curves
with the same configuration, seed, and codeword count, changing only
`--speed-mps`. The curve output retains the per-codeword CRC mask. Compare them
with the paired bootstrap and exact McNemar analysis:

```bash
python scripts/compare_tdl_pusch_time_curves.py \
  outputs/tdl-pusch-time-control.json \
  outputs/tdl-pusch-time-treatment.json \
  outputs/tdl-pusch-time-paired-comparison.json
```

The comparator verifies matching bits, batch seeds, noise seeds, and all
configuration fields other than mobility before calculating paired outcomes.
Its uncertainty is conditional on the chosen simulation; it does not replace
independent-scene inference.

To compare additional PUSCH DMRS positions at fixed mobility, use identical
SNRs, frame count, batch size, and seed, then vary only
`--dmrs-additional-position`. The comparator checks that channel-truth hashes,
batch seeds, and noise seeds match; the transmitted transport-block sizes may
differ because extra DMRS symbols consume resources. It therefore reports both
paired BLER and successfully decoded transport-block bits per codeword, so the
pilot-density tradeoff includes payload goodput:

```bash
python scripts/run_tdl_pusch_time_bler_curve.py outputs/dmrs0.json \
  --frames 512 --batch-size 64 --snr-db 12 18 24 --seed 42100 \
  --speed-mps 30 --dmrs-additional-position 0
python scripts/run_tdl_pusch_time_bler_curve.py outputs/dmrs3.json \
  --frames 512 --batch-size 64 --snr-db 12 18 24 --seed 42100 \
  --speed-mps 30 --dmrs-additional-position 3
python scripts/compare_tdl_pusch_time_curves.py outputs/dmrs0.json \
  outputs/dmrs3.json outputs/dmrs-paired.json \
  --varying-config-field dmrs_additional_position
```

This remains a single-configuration TDL-A simulation with ideal synchronization
and codeword-conditional uncertainty; it does not establish scene-level or
standards-wide receiver performance.

To compare the built-in Sionna LS estimator with the finite-delay DFT estimator,
use the same channel, seed, batch size, and SNR grid. The v2 curves retain each
codeword's batch-seed cluster ID; the comparator verifies matched payload,
channel, noise, and cluster rosters, then reports a paired cluster bootstrap at
each SNR. Use enough independently seeded batches for the intended uncertainty
analysis, and choose DFT delay bounds that match the declared channel support.
This is a bounded simulation comparison, not a G5 acceptance result:

```bash
python scripts/run_tdl_pusch_time_bler_curve.py outputs/tdl-ls.json \
  --frames 512 --batch-size 64 --snr-db 0 6 12 18 24 --seed 43000 \
  --profile A --n-size-bwp 12 --speed-mps 3 \
  --channel-estimator sionna_ls
python scripts/run_tdl_pusch_time_bler_curve.py outputs/tdl-dft.json \
  --frames 512 --batch-size 64 --snr-db 0 6 12 18 24 --seed 43000 \
  --profile A --n-size-bwp 12 --speed-mps 3 \
  --channel-estimator finite_delay_dft \
  --estimator-min-delay-samples -6 --estimator-max-delay-samples 7
python scripts/compare_tdl_pusch_time_curves.py \
  outputs/tdl-ls.json outputs/tdl-dft.json outputs/tdl-estimator-comparison.json \
  --varying-config-field channel_estimator
```

To isolate multipath delay relative to the cyclic prefix while holding the NR
PUSCH allocation and payload fixed, run the deterministic two-path diagnostic:

```bash
python scripts/run_pusch_cp_margin_experiment.py \
  docs/research/results/pusch-static-cp-margin-sweep-12prb-30khz-512.json \
  --frames 512 --batch-size 64 --seed 82417 --snr-db 30 \
  --n-size-bwp 12 --subcarrier-spacing-khz 30 \
  --in-cp-guard-samples 5 --beyond-cp-samples 3 \
  --additional-echo-delays-samples 1 4 11 13 20
```

The shortest CP is 13 samples. The default reference places the echo at 8
samples (5 samples of CP margin); the paired beyond-CP condition places it at
16. Additional requested
delays are evaluated on the same transmitted waveforms and standardized noise,
with paired CRC comparisons against the 8-sample reference. The artifact
records transmit/receive waveform power and per-batch measured receive SNR for
each delay. The path gains are deterministic and the burst starts with zero
channel prehistory, so this is a controlled ISI diagnostic, not a TDL/CDL
distribution or standards-wide receiver result.

Use `run_identifiability_sweep.py` for a local two-path Fisher-information diagnostic across tone bandwidth, SNR, and delay separation:

```bash
python scripts/run_identifiability_sweep.py outputs/identifiability-sweep.json \
  --tone-counts 256 1024 --tone-spacing-hz 30000 \
  --snr-db 0 10 20 30 \
  --separation-fractions 0 0.001 0.01 0.025 0.05 0.1 0.25 0.5 1 2
```

Separations are fractions of reciprocal sampled bandwidth. The output records
rank, normalized conditioning, per-path delay CRLB standard deviations when
finite, package/source hashes, and runtime versions. Rank-deficient points
omit covariance rather than implying finite certainty. This is a local bound
under a two-path complex-CFR plus circular-Gaussian-noise model; it is not an
estimator result, global resolution threshold, receiver experiment, or scene
validation.

Use `run_rt_solver_convergence.py` to compare deterministic CPU/LLVM Sionna RT ray budgets and interaction depths on a fixed link:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  python scripts/run_rt_solver_convergence.py outputs/rt-convergence.json \
  --scene box_one_screen --tx -2 0 2.5 --rx 0 0 2.5 \
  --max-depth 3 --samples-per-src 200 2000 20000 100000 \
  --seeds 11 23 47 --n-subcarriers 256 \
  --subcarrier-spacing-hz 30000 --carrier-hz 3500000000
```

The runner compares complex CFRs against the largest requested ray budget at
maximum depth for the same seed, and records valid paths, path powers/delays,
scene asset hashes, runtime and code provenance. The highest budget is a
reference, not assumed truth. A single built-in scene and SISO link do not
qualify convergence for other geometries or establish material realism.

For signed release splits that include the immutable UMCD episode store, use
child-artifact roster v5 with the existing `audit_channel_split_lineage.py`
entry point. The audit traverses every declared episode manifest and content
object under the exact artifact root, enforces a conservative 512 MiB
shard-accounting ceiling, and binds each episode ID to its frozen split parent.
Content objects use `child_ids` to declare every episode manifest that
references the object; the auditor requires exact owner coverage across the
whole roster. The report advances to lineage bundle v9. The accounting model
is not measured RSS, and these input-scoped reports do not establish external
inventory authority or release qualification. Roster v4 remains readable with
its original one-child content-object binding.

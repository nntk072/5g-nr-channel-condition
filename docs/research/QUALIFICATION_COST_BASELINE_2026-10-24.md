# G3 qualification compute and storage baseline

**Status:** historical planning baseline only. This is not a run manifest, a complete cost estimate, or authorization to start qualification or production RT generation.

The implementation now joins material, complex-CSI, propagation, and receiver
assurance plans against one frozen family and reports per-stratum counts and
component digests while preserving distinct sampling units. A complete design
flag means only that the explicit scenario assumptions meet the selected
cross-branch assurance target and angular constraint. The result remains
scenario-conditional and `NOT_QUALIFIED`, with no combined sample-size claim;
the historical zero-failure floor below is not upgraded by this code.

## Historical CPU measurements

The table summarizes depth-3 LLVM CPU run times in the saved 100k-versus-1M samples/source diagnostics. Each source report contains 120 runs per budget: six links × 20 seeds. Medians and 95th percentiles below are per RT run, not per candidate/challenger pair.

| Procedural scene | 100k median / p95 (s) | 1M median / p95 (s) | Pair median / sum of per-budget p95 (s) |
|---|---:|---:|---:|
| City41 | 0.0871 / 0.1464 | 0.2299 / 0.3137 | 0.3169 / 0.4601 |
| City42 | 0.0846 / 0.1161 | 0.2215 / 0.2807 | 0.3061 / 0.3968 |
| City43 | 0.0794 / 0.1157 | 0.2027 / 0.2851 | 0.2820 / 0.4009 |

Inputs (SHA-256):

- [City41 report](results/rt-city41-spatial-budget-depth3-native-materials-6links-100k-1m-20seeds-20261022.json): `b927b3af15c137220b81ed963ada7bcc6178234274798eee7686ed32695907c3`
- [City42 report](results/rt-city42-spatial-budget-depth3-native-materials-6links-100k-1m-20seeds-20261022.json): `c396f18c1a6399f6a1eb1d46f08fb3c6bc9d88853b51672e5c73f0565a207fe2`
- [City43 report](results/rt-city43-spatial-budget-depth3-native-materials-6links-100k-1m-20seeds-20261022.json.gz): `1974c27116cc0b817d05f48734915987258782cf9a6b83c12ab0fcda0b6104e2`

The prior G3 contract states that at least 59 independent outer clusters in each of three scene families × LOS/NLOS strata are needed for a one-sided 95% exact-binomial upper bound of 0.05 when zero failures are observed. That is a **zero-failure mathematical floor**, not a powered design and not enough for the other CSI, delay/angular, path-equivalence, material, or receiver criteria: 6 strata × 59 = 354 clusters. City41/42/43 share one procedural family, so two other independent families remain necessary.

## Joint assurance composition update (2026-10-25)

The acquisition-design CLI now composes evaluator-aligned material, complex-CSI,
propagation, and receiver assurance plans, then applies a cross-branch union
bound to their selected lower bounds. A study-level 0.80 assurance target is
not met merely because each branch separately meets a 0.80 target: under the
union bound, four 0.80 branch bounds imply only a 0.20 lower bound for their
joint pass event. The planner therefore reports the required branch counts and
keeps the complete-design flag false until the four branch bounds jointly meet
the study target, with material angular maximum compatibility also satisfied.
Path equivalence is counted as a census of frozen episodes and does not make a
population-generalization claim. The result remains scenario-conditional and
`NOT_QUALIFIED`; the runtime estimates above do not change.

For scale only, assume exactly one depth-3 candidate/challenger link pair per independent cluster, with 100k and 1M runs. Applying the historical median pair time gives about 107 seconds of solver time for 354 pairs; summing each scene's per-budget p95 gives about 149 seconds. Scaling to six links per cluster gives about 11 and 15 minutes, respectively. The largest observed per-budget maxima occurred in City41; summing those maxima gives a deliberately conservative historical envelope of about 13.5 minutes for one link per cluster or 81 minutes for six. These are arithmetic projections from small procedural scenes, not forecasts for the OSM scenes or nonzero-scattering configurations.

## Limits and unresolved cost

All three source campaigns used procedural City scenes, LLVM CPU, depth 3, and native concrete/glass/metal scattering coefficients of zero. The reports do not record a CPU model or controlled host-load measurement, so runtimes are not portable benchmarks. They do not measure the intended distinct scene families, calibrated nonzero scattering, full 3D path-lineage retention, waveform/receiver scoring, or storage for production outputs. Their reports omit the full path arrays and CFR vectors; report-file size is therefore not a valid estimate of raw corpus size. OSM scene complexity, path counts, waveform views per episode, retained raw artifacts, compression, and shard/manifest overhead remain unmeasured.

Consequently this baseline is useful only for rejecting a claim that the historical small-scene solver loop is intrinsically hours-long. Before a confirmatory campaign, freeze the exact scene/link/episode design, estimate the number and size of all retained artifacts, and obtain a bounded cost measurement on an authorized representative target scene and material configuration. That pilot remains exploratory and must not be used to tune or qualify the same held-out data.

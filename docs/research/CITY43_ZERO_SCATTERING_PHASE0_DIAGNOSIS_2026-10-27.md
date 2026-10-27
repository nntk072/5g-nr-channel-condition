# City43 zero-scattering LOS instability: bounded Phase-0 diagnosis

**Disposition:** Existing path-level diagnostics support budget-dependent discovery/multiplicity of identical zero-interaction path records, followed by coherent accumulation, as the dominant explanation for this one observed CFR change. They do not establish a physical channel, convergence, or a general solver defect.

## Scope and method

This is a read-only synthesis of retained path-level results. It uses the City43 `center-los` SISO link, seed 23, 3.5 GHz, zero scattering, maximum interaction depth 10, fixed scene and receiver/transmitter geometry, and the existing 100,000-versus-1,000,000 samples-per-source reports. Both reports use the LLVM CPU backend and report that the candidate path cap was not filled. The larger finite budget is a comparison condition, not ground truth.

The interpretation uses the path-lineage reports to separate records generated after the LOS stage, records retained after invalid-path removal, and their interaction orders. It uses the path-match sensitivity report to compare matched nonzero-interaction coefficients and geometry, then aggregate the zero-interaction class separately. Fingerprints are scene/link-local matching aids, not universal physical path identities.

## Evidence

| Diagnostic | 100k | 1M |
| --- | ---: | ---: |
| Candidate records before path validation | 6,831 | 29,110 |
| Valid records after invalid-path removal | 194 | 597 |
| Valid zero-interaction records | 9 | 75 |
| Zero-interaction records with zero TX angle | 8 | 74 |
| Unique zero-interaction coefficients at 1e-12 resolution | 1 | 1 |
| Unique zero-interaction delays at 1 ps resolution | 1 | 1 |
| Unique zero-interaction geometry fingerprints | 1 | 1 |
| Coherent zero-interaction component mean power | 1.964e-6 | 1.364e-4 |
| Total CFR mean power | 2.323e-6 | 1.393e-4 |

The sensitivity audit attributes **0.999949** of the total CFR-delta norm to the aggregate zero-interaction class. Its count grows from 9 to 75 while all records collapse to one coefficient, delay, and geometry fingerprint at the report's stated resolutions. This is consistent with coherent accumulation of repeated zero-interaction records discovered as the candidate budget changes.

The nonzero-interaction paths show substantial candidate-roster growth and different interaction-order counts: order 10 grows from 15 to 105 valid paths, and orders 8–9 also grow. This is a secondary budget-sensitive discovery effect in these diagnostics. At order 10, however, the reported coherent component power is 1.39e-14 at 100k and 2.31e-12 at 1M, far below the corresponding zero-interaction component. Interaction-order growth therefore does not explain most of the total CFR change in this case.

For matched nonzero-interaction paths, the reported maximum interaction-vertex displacement is below 0.8 mm, maximum absolute-delay difference is about 0.000114 ns, maximum direction difference is below 1.4e-5 rad, and maximum symmetric complex-coefficient relative error is about 7.9e-5. These values do not support coefficient or geometry instability among the matched paths as the main cause. The matching method cannot prove that unmatched records are physically distinct.

## Bounded conclusion

For this saved configuration, the evidence points to **path discovery/multiplicity of zero-interaction records feeding coherent accumulation**. Higher-order path discovery changes too, but its reported power is small relative to the zero-interaction aggregate. Matched-path coefficient and geometry changes are small at the reported precision. The evidence does not identify why Sionna RT returns the extra zero-interaction records, prove that they are physically invalid, or show how another seed, geometry, budget, backend, or scattering model behaves.

No new convergence experiment, seed sweep, ray-budget escalation, or solver intervention was run. The 1M result is not declared ground truth. Production RT generation and G3 acceptance remain gated by their separate criteria.

## Retained sources

- [Path-lineage report, 100k vs 1M](results/rt-city43-center-los-seed23-depth10-s0-path-lineage-100k-vs-1m-20261023.json.gz) — SHA-256 `0ccd0484eec1481443d2a8fabea3517b171cc76d234665307c3678028b274844`.
- [Path-lineage write-hook replay, 100k vs 1M](results/rt-city43-center-los-seed23-depth10-s0-path-lineage-writehook3-100k-vs-1m-20261023.json.gz) — SHA-256 `89133167eab4338d00e67b24c23bb81044bffa13a1a13160bbc1c705723d1093`.
- [Path-match threshold sensitivity](results/rt-city43-center-los-seed23-depth10-s0-path-match-sensitivity-20261023.json) — SHA-256 `37e40dc3fa6541b259d919bb72f4e26334683027775235e299fd6a8c28820187`.
- [Original path-attribution audit](results/rt-city43-center-los-seed23-depth10-s0-path-attribution-audit-20261022.json) — SHA-256 `da134629834e5e37e3760392944f24876d5acef3aed14651561216cedec91e81`.

The source reports carry their own scene-tree, XML, package-source, script, and runtime identities. This summary is limited to the evidence captured in those exact artifacts.

# Research plan topic: qualification gates and evidence

This note records the experiments, metrics, statistical rules, stress tests,
validation levels and stop conditions from the archived master plan. It states
what evidence must exist before a result is called qualified.

**Source:** the archived plan at git tag `docs/master-research-plan-archive`
(file `docs/research/MASTER_RESEARCH_PLAN.md`), sections 7, 9, 10, 15 and
Appendix E.3. Current gate state is in [MVP_STATUS.md](MVP_STATUS.md) and in the
requirement ledger, not in this note.

## Experiment catalogue

The plan lists 24 experiments, E00 to E24, each with a question, a minimum
comparison, its outputs and a decision gate. They cover the reproduction of the
2025 results, analytical and native channel checks, RT oracle fidelity and
convergence, receiver correctness, profile leakage, LOS balance, the MLP versus
CNN contribution, the identifiability ablation (E11), phase, timing and array
gauge tests, source-to-target transfer, real measurements, counterfactual
blockage, Doppler, blockage forecasting, beam impact, calibration, robustness,
cost, data scaling, cross-simulator validity and final replication.

Every run records an immutable run identifier, the input, scene, split and source
hashes, the label definition, the exact environment, the checkpoints, the
predictions, the scalar and per-group metrics, the resource cost and the failures.
Negative results and failed runs are kept.

## Split protocols

The plan names nine evaluation protocols, P0 to P8, and says what each proves.
A random split of the same generator (P0) is a debugging check and is not
scene transfer. Only the held-out scene (P4), the held-out material or frequency
(P5), the complete held-out trajectory (P6) and the frozen measured site (P7)
support a transfer claim. Few-shot adaptation (P8) is not zero-shot. No test
label may affect the normalization, selection, calibration, early stopping or
thresholds.

## Statistical rules

The primary unit for external validity is the scene, site or session, then the
trajectory within it. Ten random seeds on one city are not ten cities. Paired
seeds, splits and budgets are required. Multiple contrasts are controlled, for
example by Holm correction, and effect sizes are reported with intervals. The
prevalence of the raw world and of any balanced diagnostic is reported
separately. The success criteria are set before training.

## Falsification tests

Thirty stress tests, S01 to S30, are listed. Each states what should stay the
same and what should change if the claim holds. Examples include a common phase
rotation, a common time shift, path-index permutation, a weak direct path pushed
below noise, a blocker inserted with a full scene re-trace, shuffled future
samples, a label-permutation negative control, and a shuffled-metadata check that
the model is not reading scene or label information.

## Validation levels

The reproducibility levels run from L0 (pure numerical and schema tests) to L7
(a second environment regenerates every claimed table and figure from immutable
results). A level is entered only when the level below passes. A Sionna version
mismatch fails loudly, and a native contract that is not run on the host is not
counted as passed.

## Severity-one failures

A failure of any of these invalidates the claim that depends on it, and the
experiment stops. Examples: profile identity used as geometric LOS, absolute delay
reported from normalized delays, unobservable truth or test statistics among the
classifier inputs, the same scene or trajectory in training and test, silent
discarding of outages, and a result published without a new independent
evaluation after a test-scene selection.

Any change to a unit, a convention, a mask, a target, a material default, a
Sionna flag, a pilot grid, a split identifier, a noise definition or a metric
aggregation needs a changelog entry, a new contract version, regenerated results
under new hashes, and a comparison with the frozen baseline. Results are never
patched in place.

## Stop and go

The phase gates run P0 to P11. The plan states that the project remains
simulation-only software until the RT contracts, an end-to-end synchronized NR
receiver test, cross-environment datasets and a novelty ablation exist. Acceptance
is decided from evidence, never from ambition. The final charter lists eight
conditions, ending with an unbroken chain of provenance for every accepted number.

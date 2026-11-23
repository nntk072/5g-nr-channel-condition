# Research plan topic: open questions and risks

This note records the open questions, the risk register, the pivots the plan
allows, and the reviewer checks that the plan asks for. It lists questions, not
answers.

**Source:** the archived plan at git tag `docs/master-research-plan-archive`
(file `docs/research/MASTER_RESEARCH_PLAN.md`), sections 0.3, 11, 11.1, 14.3 and
Appendix E. The archived plan's status tables are historical; the current ledger
is in [MVP_STATUS.md](MVP_STATUS.md).

## Open questions

1. **Novelty.** Does the proposed identifiability-aware mechanism beat the closest published multipath, CVNN, ToA multitask and symmetry methods? The plan requires a living related-work matrix and a precise mechanism before a title claim.
2. **Advantage over the MLP.** Does any CNN or CVNN contribution survive matched inputs, augmentation, parameter and FLOP envelopes and tuning budget?
3. **Observability.** Which path parameters are identifiable from the actual pilot pattern, bandwidth, array and time window? Which need an independent reference?
4. **Transfer.** Does a model trained on standardized profiles or simulated scenes transfer to independent RT scenes and to measured data, or does the standardized pretraining cause negative transfer?
5. **Calibration.** Are the uncertainty intervals calibrated on new scenes, materials, bandwidths and hardware, and where do they fail?
6. **Forecasting.** Can blockage and path birth and death be forecast with time-updated geometry, and does a forecast help a downstream beam or receiver decision?
7. **Real data.** Is there permission-cleared, calibrated, phase-sensitive measured CSI with independent surveyed truth? The archive records that only magnitude-only exploratory data and a lab acquisition proposal existed when it was written.
8. **Runtime.** Do the native Sionna RT and PHY pins run on the production host? The archive records a compatibility problem with WSL and with an OptiX-based runtime, and it asks for a separate compatible backend rather than a silent switch.

## Risk register

The archived register has 27 risks, each with a severity, a detection method and
a mitigation or pivot. The highest-severity entries are:

- Geometric LOS mistaken for a profile label.
- Absolute delay lost through normalized delays.
- The Sionna 2.2 API differing from the mocked contract.
- Wrong phase or baseband convention.
- Counterfactual blockage made by deleting one coefficient.
- Test leakage through near-identical geometry.
- Timing labels shifted by clock drift.
- Negative results hidden.

## Pivots the plan allows

- No robust direct-path detection: focus on calibrated channel-state uncertainty, and state that LOS is not identifiable there.
- MLP dominates the CVNN: use the MLP as the encoder and contribute the physical decoder or loss.
- Exact per-ray counts or angles do not generalize: predict the band-limited PDP, dispersion and resolvable clusters with uncertainty.
- Standardized pretraining harms RT transfer: quantify the negative transfer and do not force the standardized model into the pipeline.
- No real data: publish a rigorous synthetic benchmark with explicit external-validity limits, and do not use the words "field-validated".
- Dynamic RT too expensive: make the first paper a static identifiability paper and treat forecasting as a separate project.
- Incremental novelty after review: position the work as an empirical benchmark study.

## Reviewer checks

Before submission the plan asks for six adversarial checks: whether a conventional
LS or LMMSE estimator already solves the task, whether the MLP gains persist
under matched budgets, whether the endpoints and splits were locked, whether the
parameters are identifiable and the intervals calibrated, whether the result
reproduces from immutable hashes, and whether a better parameter estimate gives a
useful decision.

## Prior art to verify

The archived plan lists standards and publications to check before any citation.
It states that the paper-specific details had not been verified at the time the
list was written, so each reference needs a check against its source before use.

# G5.09 Shared-Synchronization Diagnostic-Retention v3 Protocol

**Status:** Frozen before implementation of the additive retention fields and
before the run on 2026-10-25.

This bounded third independent-seed replication keeps the existing shared-CP
receiver experiment and all seven acceptance thresholds unchanged. Its sole
implementation change is to retain the already-computed per-channel
samplewise-reference maximum absolute residual and wrong-order waveform RMSE,
then independently recompute the experiment-wide maximum/minimum from those
retained scalar records in the result auditor. Raw waveforms are not added, so
this improves aggregation auditability but does not independently re-derive
the waveform equations.

## Frozen run

- Experiment: shared-CP timing/CFO correction for the same 6-RB/30-kHz CPU
  PUSCH and synthetic two-tap channel family as the first two protocols.
- Channel seeds: `160000` through `160063` inclusive; 64 independent channel
  draws and eight paired payload blocks per channel. A structured scan of all
  229 saved JSON/JSON.GZ result reports found no seed-key collision in this
  roster. Earlier rosters are `74001`–`74064`, `80000`–`80063`, and
  `120000`–`120063`.
- Channel law, impairment values/order, shared synchronization estimators,
  fixed timing/CFO guards, nested-channel bootstrap, and all receiver behavior
  remain unchanged. The runner records two extra scalar fields per channel;
  the auditor recomputes their aggregate maximum/minimum from these fields.
- Compute bound: exactly 64 channel seeds × 8 blocks; no retries, tuning,
  budget sweeps, or replacement seeds.

## Frozen acceptance criteria

All seven existing criteria are unchanged and must pass:

1. Maximum absolute timing acquisition error is exactly 0 samples.
2. P95 absolute CFO estimation error is strictly below 250 Hz.
3. Maximum absolute CFO estimation error is strictly below 350 Hz.
4. Maximum samplewise-reference residual is strictly below `2e-5`.
5. Minimum wrong-order waveform RMSE is strictly above `1e-4`.
6. Clean-reference BLER is at most 0.01.
7. Corrected-impaired BLER is at most 0.10.

The 20,000-draw paired channel-cluster interval and exact CFO-tail diagnostics
remain descriptive, not additional pass conditions. The retained-result audit
must additionally prove that experiment-level residual/RMSE summaries and the
pass verdict agree with the per-channel scalar records. No criterion may be
changed after observing the run.

## Run and disposition

Run once with the implemented v3 runner using channel seed start `160000`, 64
independent channels, and eight blocks per channel. Preserve any failing result
unchanged; do not rerun these seeds. A pass remains evidence for this one
declared synthetic law and does not establish field/deployment receiver
performance, G5.09 closure, production RT authorization, or advanced-training
authorization. Raw waveform equations remain unindependently recomputed.

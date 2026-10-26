# G5.09 Shared-Synchronization Independent Replication Protocol

**Status:** Frozen before the replication run on 2026-10-25.

This protocol independently replicates the retained 64-channel shared-CP
synchronization experiment using a new, non-overlapping channel seed roster. It
uses the existing implementation and acceptance criteria unchanged. The
replication tests repeatability across channel draws; it does not broaden the
synthetic channel law into a field population or close G5.09.

## Frozen experiment

- Runner: `experiments/g5_ordered_impairments_receiver.py`, independent-channel
  probe, exact source bytes recorded in the result provenance.
- Runtime: locked Python environment, Sionna PUSCH PHY on CPU; no geometric RT
  solver or GPU.
- Channel units: 64 independent two-tap channel seeds, `80000` through `80063`
  inclusive, eight paired payload blocks nested within each channel.
- Data-use check: the proposed channel seeds were scanned against the 222 saved
  JSON/JSON.GZ research result reports before this protocol was frozen; none
  appear there. The earlier shared-sync confirmation uses seeds `74001`–`74064`.
- Waveform: 6-RB, 30-kHz PUSCH; direct coefficient `0.9 - 0.1j`; echo amplitude
  uniform on `[0.20, 0.50]`, phase uniform on `[-π, π]`, and integer delay
  uniform on `[2, 6]` samples. Each channel uses one draw from this declared law.
- Impairments: TX and RX Wiener phase noise with 10-Hz linewidth each, fixed
  TX/RX IQ gain imbalance 0.5 dB and phase imbalance +0.02/-0.02 rad, 600-Hz RX
  CFO, two-sample capture timing offset, and proper complex AWGN of variance
  `1e-3`. The physical order remains TX IQ/phase noise → static multipath → RX
  CFO/phase noise/IQ imbalance → timing offset → AWGN.
- Synchronization: one shared CP-based timing/CFO estimate per channel, with
  fixed timing guard 2 samples and CFO guard 6 samples. Neither guard may use
  the realized echo delay. Correction uses the acquired timing and CFO.
- Controls: clean channel with timing alignment; actual impairment chain with
  no synchronization correction; and the actual chain with shared timing/CFO
  correction. A separately spelled sample-reference chain and an intentionally
  wrong oscillator/channel ordering remain as implementation controls.
- Compute bound: exactly 64 channel seeds × 8 blocks; no retries, budget sweeps,
  parameter tuning, or additional channel draws are permitted for this run.

## Frozen acceptance criteria

The unchanged runner criteria must all pass:

1. Maximum absolute timing acquisition error is exactly 0 samples.
2. P95 absolute CFO estimation error is strictly below 250 Hz.
3. Maximum absolute CFO estimation error is strictly below 350 Hz.
4. Maximum samplewise residual against the separately spelled reference is
   strictly below `2e-5`.
5. Minimum wrong-order waveform RMSE is strictly above `1e-4`.
6. Clean-reference BLER is at most 0.01.
7. Corrected-impaired BLER is at most 0.10.

P95 is the nearest-rank statistic in the runner. Corrected-minus-clean BLER
uses a paired bootstrap resampling the 64 channel clusters, with the frozen
20,000 draws and bootstrap seed `74999`; the eight blocks within each channel
are not treated as independent clusters. CFO tail diagnostics use exact
one-sided 95% binomial upper bounds on the fraction of channels with any block
exceedance at the 250-Hz and 350-Hz thresholds. These inference summaries are
reported as descriptive uncertainty diagnostics; they add no post-run pass
criterion.

## Run and disposition

Run exactly once:

```bash
uv run --no-sync --locked python -m experiments.g5_ordered_impairments_receiver \
  --independent-channel-count 64 \
  --blocks-per-seed 8 \
  --channel-seed-start 80000 \
  --output docs/research/results/g5-ordered-impairments-independent-channels-cpu-64x8-shared-sync-replication-20261025.json
```

Run the result auditor against the retained report without deterministic
replay. Preserve a failed result unchanged; do not rerun these seeds, revise
criteria, or select a passing subset. A pass is an independent-seed replication
of this one synthetic fixture only. It is not evidence of a measured channel
population, general NR deployment performance, G5.09 closure, production RT
authorization, or advanced-training authorization. The worktree is not
commit-bound until the source is committed.

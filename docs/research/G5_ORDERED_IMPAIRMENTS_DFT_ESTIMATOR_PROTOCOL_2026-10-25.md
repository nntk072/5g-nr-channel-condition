# G5 ordered-impairment confirmation with finite-delay DFT estimation

**Status:** implementation and regression definitions authored; experiment,
seed-collision audit, and consolidated validation not run. This protocol does
not change any historical result or close G5.09.

The current implementation revision also retains, for clean/corrected/uncorrected
conditions, the number of independent channels with at least one CRC failure in
the fixed repeated-block window and its exact one-sided 95% binomial upper bound.
This is a channel-event prevalence diagnostic, not a mean-BLER interval or a
new acceptance threshold. Result schemas v11–v14 bind this diagnostic; older
schemas remain readable as historical artifacts without it.

## Question

Does the finite-delay DFT estimator recover the frozen ordered-impairment
PUSCH fixture under the same independent-channel design and acceptance limits
as the existing Sionna default-LS run?

## Frozen implementation and design

- Receiver: Sionna 2.2 CPU, 6-RB BWP, 30-kHz SCS, with the existing paired clean,
  uncorrected, and corrected waveform chain.
- Channel estimator: raw-pilot finite-delay DFT fit on integer delays 0 through
  7 samples inclusive, using the implementation in
  `experiments/g5_cp_bounded_dft_estimator.py`. The runner retains its exact
  SHA-256 and the auditor checks the worktree implementation and, when supplied,
  a requested committed revision.
- Independent channels: seeds 180001 through 180064 inclusive, with eight
  repeated blocks per channel. Before launch, run the seed-roster collision
  auditor across retained G5 artifacts. Any collision invalidates this roster;
  issue a new protocol revision with a fresh contiguous range before running.
- Retain the raw waveform evidence sidecar and audit the signal equations and
  estimator source binding. Deterministic replay is a reproducibility check,
  not independent mathematical validation. The source digest identifies the
  file on disk; it does not attest the estimator object loaded in process or
  third-party Sionna binaries.

## Acceptance

Use the existing ordered-impairment thresholds unchanged: exact timing;
absolute CFO P95 below 250 Hz and maximum below 350 Hz; samplewise reference
maximum error below `2e-5`; wrong-order waveform RMSE above `1e-4`; clean BLER
at most `0.01`; corrected BLER at most `0.10`. Keep the channel as the
independent inference unit and retain every per-channel outcome and interval.

All criteria apply only to this named synthetic fixture. A pass does not
establish field BLER, CP-exceeding-channel behavior, generalized receiver
qualification, or production-RT readiness. Existing default-LS historical
reports remain immutable and are not rewritten or pooled with this result.
The point BLER criteria remain the frozen finite-fixture acceptance checks;
the channel-event bounds expose clustering and are reported separately without
changing the acceptance verdict.

## Reproduction after implementation validation

```sh
uv run --no-sync --locked python scripts/audit_seed_roster_collisions.py \
  --root docs/research/results --root experiments/results \
  --seed-range 180001 180065
uv run --no-sync --locked python experiments/g5_ordered_impairments_receiver.py \
  --independent-channel-count 64 --channel-seed-start 180001 \
  --blocks-per-seed 8 --channel-estimator finite_delay_dft \
  --output docs/research/results/g5-ordered-impairments-dft-20261025.json
uv run --no-sync --locked python experiments/audit_g5_ordered_impairments_result.py \
  docs/research/results/g5-ordered-impairments-dft-20261025.json
```

The commands above are recorded for the later evidence phase and have not been
run as part of the implementation pass.

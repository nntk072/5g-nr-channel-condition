# G5.09 Shared-Synchronization Independent Replication v2 Protocol

**Status:** Frozen before the v2 run on 2026-10-25.

This is a third seed-disjoint replication of the existing shared-CP timing/CFO
receiver fixture. It uses the same runner and acceptance criteria as the
confirmation and first replication. It does not broaden the synthetic channel
law, validate a field population, or close G5.09.

## Frozen run

- Runner: `experiments/g5_ordered_impairments_receiver.py`, independent-channel
  probe. Frozen worktree SHA-256: `81aaf06d9f3e69c580703fe59c92d80e230e4830f0edcb83dfe7c6a81e3c2b04`.
- Runtime: locked Python environment, Sionna PUSCH PHY on CPU; no RT solver or
  GPU.
- Channel units: 64 channel seeds, `120000` through `120063` inclusive, with
  eight paired payload blocks nested within each channel. A scan of all 227
  saved JSON/JSON.GZ result reports found no exact seed-key collision in this
  roster. The first replication used `80000`–`80063`; the original confirmation
  used `74001`–`74064`.
- Channel law and waveform: unchanged 6-RB/30-kHz PUSCH; direct coefficient
  `0.9 - 0.1j`; echo amplitude uniform on `[0.20, 0.50]`, phase uniform on
  `[-π, π]`, and integer delay uniform on `[2, 6]` samples.
- Impairments, physical order, shared CP-based timing/CFO acquisition, fixed
  timing/CFO guards, three receiver conditions, and nested-channel inference
  are unchanged from the first replication protocol
  (`G5_09_SHARED_SYNC_INDEPENDENT_REPLICATION_PROTOCOL_2026-10-25.md`).
- Compute bound: exactly 64 channel seeds × 8 blocks; no retries, tuning,
  budget sweeps, or replacement seeds.

## Frozen acceptance

All seven unchanged runner criteria must pass:

1. Maximum absolute timing acquisition error is exactly 0 samples.
2. P95 absolute CFO estimation error is strictly below 250 Hz.
3. Maximum absolute CFO estimation error is strictly below 350 Hz.
4. Maximum samplewise-reference residual is strictly below `2e-5`.
5. Minimum wrong-order waveform RMSE is strictly above `1e-4`.
6. Clean-reference BLER is at most 0.01.
7. Corrected-impaired BLER is at most 0.10.

The runner's fixed 20,000-draw paired channel-cluster bootstrap and CFO tail
diagnostics remain descriptive and are recomputed by the retained-result
auditor. No criterion may be changed after observing the run.

## Execution and disposition

Run exactly once:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync --locked \
  python -m experiments.g5_ordered_impairments_receiver \
  --independent-channel-count 64 \
  --blocks-per-seed 8 \
  --channel-seed-start 120000 \
  --output docs/research/results/g5-ordered-impairments-independent-channels-cpu-64x8-shared-sync-replication-v2-20261025.json
```

Audit the retained artifact against the exact worktree source. Preserve a
failure unchanged; do not rerun this seed roster. A pass is evidence for this
declared synthetic receiver fixture only. It does not authorize production RT
or advanced training. The artifact remains uncommitted until the planned
source finalization commit.

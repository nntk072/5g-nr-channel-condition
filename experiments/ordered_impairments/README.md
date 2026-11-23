# experiments.ordered_impairments

Concept modules for the ordered-impairment receiver check. The public entry points stay in `experiments/ordered_impairments_receiver.py`, which re-exports these names and holds the command-line interface and the hash-bound result writer.

## Modules

| Module | Contents |
| --- | --- |
| `constants.py` | Schema identifiers, seed rosters, frozen thresholds, byte budgets |
| `hashing.py` | Canonical JSON and SHA-256 helpers |
| `signal_models.py` | Complex noise, phase noise, and the known impairment chain |
| `waveform_evidence.py` | Bounded NPZ writer and combined SNR-curve waveform evidence |
| `statistics.py` | Cluster bootstrap, failure-probability bound, shared-sync pooling |
| `paired_probe.py` | Paired (single-channel) impairment probe and decoder selection |
| `independent_probe.py` | Independent-channel probe and its report schema |
| `independent_snr_curve.py` | Independent-channel SNR curve and its cluster inference |

## Invariants

- The producer source binding in `build_hash_bound_result` hashes the facade file only. The implementation modules are not covered by that digest; see `dev-docs/operations/refactor-verification.md`.
- A function is looked up in the module that calls it. Tests that replace a function must patch every module that binds it.
- File-relative paths in these modules use `parents[2]` for the repository root.

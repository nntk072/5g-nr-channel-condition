# experiments.audit_tdl

Audits of the bounded TDL PUSCH evidence, one module per audited protocol family.

## Purpose

Each module re-computes the statistics of one frozen protocol and checks a retained
result against it. The modules were previously separate top-level files named
`experiments/audit_tdl_*.py`. They are grouped here so the shared hash helpers live once.

## Modules

| Module | Contents |
| --- | --- |
| `hashing.py` | SHA-256 and canonical-JSON helpers shared by the audits |
| `c_dmrs_density.py` | TDL-C DM-RS density confirmation audit (`audit_report`) |
| `dft_density.py` | DFT-estimator density audit (`audit_dft_report`) |
| `dft_profile_matrix.py` | DFT profile-matrix family audit |
| `dmrs_density_development.py` | Development-stage DM-RS density audit (`run_development_audit`) |
| `profile_matrix.py` | Profile-matrix family audit (`audit_report`) |

Run a module with `python -m experiments.audit_tdl.<module>`.

## Invariants

- Retained results keep their recorded source digests. The development module's
  pre-merge digest (`156d5617…`) is recorded in `docs/reference/digest-rebindings.json`
  as superseded by its current digest. A rebinding records supersession, not equivalence.
- `ROOT` is the repository root, `parents[2]` from this package.

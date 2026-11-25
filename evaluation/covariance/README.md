# evaluation.covariance

Reference channel models used to check generated channels against 3GPP-style
statistics: CDL covariance (`cdl/`), TDL profile and covariance references,
Doppler spectrum, complex circularity, and spherical angular power.

## Invariants

- Reference values are read from `config/reference/`; a module never writes them.
- Repository-root paths are computed from each module's location (`parents[...]`).

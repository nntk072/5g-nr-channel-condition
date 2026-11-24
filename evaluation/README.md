# evaluation

Evaluation of channel-condition models: scoring, channel-reference models,
acceptance contracts, and dataset audits.

## Layout

| Subpackage or module | Purpose |
| --- | --- |
| `scoring/` | Metrics, paired statistics, uncertainty estimates, calibration, class support, prior shift, low-confidence and coverage checks |
| `covariance/` | Reference channel covariance and power models: CDL covariance (`covariance/cdl/`), TDL profile and covariance references, Doppler spectrum, circularity, angular power |
| `acceptance/` | Contracts for accepting evaluation results: joint acceptance, qualification contracts, receiver and propagation assurance, path-equivalence checks, seed rosters, attestations |
| `audits/` | Dataset audits: duplicates, near-duplicates, population coverage, spatial leakage |
| `sample_size/`, `run_manifest/`, `acquisition/` | Sample-size planning, run manifests with source binding, and acquisition packages |
| `*_planning.py`, `joint_evidence_*.py`, `acquisition_package.py` | Public facades for the acceptance workflow |

Modules import each other by their full dotted path, for example
`evaluation.scoring.metrics`. Subpackage boundaries are checked by the
architecture tests.

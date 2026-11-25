# evaluation.acceptance

Contracts that decide whether an evaluation result is accepted: joint acceptance
families, qualification contracts, receiver and propagation assurance, path
equivalence, seed rosters, attestations, and the complex-CSI and material
calibration checks they depend on.

## Invariants

- Acceptance never changes a result's status. It reports the status the evidence
  supports and names the checks that failed.
- Contract versions are part of the output schema; changing one needs a new version.

# evaluation

**Purpose.** Metrics, statistics, reference covariance models, acceptance
contracts, sample-size planning, run manifests, and dataset audits used to judge
channel-condition models and datasets.

**Public API (by subpackage).**

- `evaluation.scoring` — metrics, paired statistics, uncertainty intervals,
  calibration, class support, prior shift, and coverage checks.
- `evaluation.covariance` — reference channel covariance and power models
  (CDL and TDL covariance, Doppler spectrum, circularity, angular power).
- `evaluation.acceptance` — joint acceptance families, qualification contracts,
  receiver and propagation assurance, path-equivalence checks, seed rosters, and
  signed attestations.
- `evaluation.audits` — dataset audits: duplicates, near-duplicates, population
  coverage, and spatial leakage.
- `evaluation.sample_size`, `evaluation.run_manifest`, `evaluation.acquisition`
  — sample-size planning, run manifests bound to their source, and acquisition
  packages.
- `evaluation.joint_evidence_artifact` — the `loslab-joint-evidence-artifact`
  command.

**Invariants.**

- An acceptance result never qualifies a model, scene, or dataset by itself.
  Each contract states its own scope, and its output records the status it
  does and does not establish.
- Multiplicity is controlled across the whole joint family. A separate claim
  cannot be tested at a lower correction than its family.
- Seeds, rosters and source bundles are part of the identity of a result.
  Changing one changes the identity.
- Signature and trust-file checks fail closed. Unsigned or unverifiable inputs
  are reported as diagnostic only.

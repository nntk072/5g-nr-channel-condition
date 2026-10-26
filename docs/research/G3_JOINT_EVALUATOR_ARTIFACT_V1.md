# G3 Five-Branch Numerical Evaluator Artifact v1

This file contract runs and replays the five deterministic numerical branches
under one acceptance-family-v4 input:

- material calibration;
- complex CSI;
- propagation metrics;
- receiver impact; and
- path equivalence diagnostics.

It makes the existing in-memory scorer usable with retained files. The separate
`loslab.g3_joint_evidence_run_manifest.v2` contract binds the exact input and
artifact paths, lengths, hashes, and roles, then verifies and replays the
captured bytes. Its verifier is available as
`evaluation.g3_joint_evidence_run_manifest.verify_g3_joint_evidence_run_manifest`.
The manifest also binds the complete fixed roster of evaluator source modules,
the shared bounded-file reader, the signed-attestation verifier, and all
input/artifact/manifest/contract schemas, plus `pyproject.toml` and `uv.lock`;
it rejects missing, additional, or changed dependency files. The source
bundle digest covers the ordered path/size/hash records, including declared and
locked Python dependencies. It does not attest the Python runtime's loaded code
objects, installed package set, or third-party dependency binaries.
The current roster also pins the path-equivalence comparator and the propagation
and receiver assurance planners; the historical v1 roster remains frozen. A
source digest makes these implementations identifiable but does not prove that
the supplied raw path records came from the declared solver execution.
Source reads are capped at 64 MiB in aggregate, and each evaluator evidence
file is capped at 32 MiB when captured by the run-manifest verifier.
It does not extend the G3.03 acquisition package, authenticate measurement
provenance, establish unit independence, or qualify a scene. Every artifact is
permanently marked `NOT_QUALIFIED`.

## Input encoding

Diagnostic v1 inputs follow
`datasets/schema/g3-joint-evaluator-input.v1.schema.json`. Input v2 adds
candidate/challenger configurations, waveform, link budget, and a digest of
that declared execution context on every receiver-impact row. Input v3 adds
the receiver row's scene, condition, and episode identity. New contract-bound
runs require v4: each receiver row also names its `configuration_id` and
includes `receiver_configuration_sha256`, which must match the exact candidate
or challenger configuration frozen in the contract. The preflight checks this
before running scorers, and replay checks it again. Input v5 carries
solver-valid deterministic/specular candidate and challenger paths, their full
active-grid CFR vectors, and one shared path-comparison configuration. The
evaluator builds effective classes and recomputes matches, coverage, and
receiver relevance; v5 rejects supplied comparison objects and unbound raw
record fields. Contract v4 freezes the matching tolerances, channel floor, and
per-RE transmit/noise powers. New contract-bound runs require this
contract-v4/input-v5 pairing; contract-v3/input-v4 is retained for historical
diagnostic replay. Inputs v1-v4 remain available for diagnostic replay, not new
contract-v4-bound runs. JSON must be
UTF-8, sorted by object key, compact, finite, and terminated by one newline.
Complex scalars use `{"$complex":[real,imaginary]}`; a complex vector is an
array of these values. Numeric vectors and matrices use JSON arrays. The parser
rejects duplicate keys, non-standard constants, noncanonical bytes, linked
files, changing files, and inputs larger than 256 MiB.

For contract-v3/input-v4 diagnostic replay, each path-equivalence comparison
declares its candidate/challenger configuration IDs, delay tolerance, and
bidirectional coverage criterion. Preflight checks those declarations against
the contract. Contract-v4/input-v5 instead recomputes the comparison from raw
path and CFR records using frozen settings. Both paths remain diagnostic-only.
V5 currently handles deterministic/specular effective-path classes;
stochastic diffuse paths require distributional comparison and are not treated
as individually identifiable rays.

The material-calibration branch embeds the same acceptance-family-v4 object
as the root. Every branch output is recomputed by the current scorer
implementation. For path equivalence, v5 recomputes from raw inputs and records
a digest for each episode's path/CFR payload; older input versions recompute
their roster summary from supplied comparison reports. The artifact records
the exact input-byte SHA-256 and a digest of the canonical serialized bundle.

## Run and replay

```bash
python -m evaluation.g3_joint_evidence_artifact evaluate joint-input.json joint-report.json
python -m evaluation.g3_joint_evidence_artifact verify joint-input.json joint-report.json
```

The evaluator refuses to overwrite an existing report. Replay checks the exact
input bytes, canonical artifact encoding, all five branch outputs, and every
recorded digest. Its pass status means only that the report matches a
deterministic recomputation under the current worktree.

## Run-manifest contract

New standalone manifests follow
`datasets/schema/g3-joint-evidence-run-manifest.v2.schema.json`; the verifier
retains the v1 schema and source-roster reader for historical receipts. A v2
manifest is canonical
compact UTF-8 JSON with one trailing newline. The two distinct relative paths
must resolve to regular files below the selected package root; symlinks and
path traversal are rejected. Hashes and byte lengths are checked before the
same captured bytes are replayed. Captured JSON files are capped at 32 MiB to
match the acquisition verifier's bounded artifact reader.

Run the manifest verifier from the repository root:

```bash
python -m evaluation.g3_joint_evidence_run_manifest run-manifest.json
```

An authorized custodian can sign the exact manifest bytes with OpenSSH in the
dedicated namespace and verify it against a separately governed trust file:

```bash
ssh-keygen -Y sign -n loslab-g3-joint-run-manifest-v1@github.com \
  -f "$G3_RUN_SIGNING_KEY" run-manifest.json
python -m evaluation.g3_joint_evidence_run_manifest run-manifest.json \
  --signature run-manifest.json.sig \
  --trusted-allowed-signers /etc/loslab/g3-allowed-signers \
  --signer-id g3-reviewer@example.org
```

The verifier reports signature, signer, namespace, signature-file, and trust
file hashes. A valid signature authenticates the manifest bytes under the
supplied allowed-signers file. The verifier cannot establish who governs that
trust file or whether the signed scientific claims are true; run receipts
remain `NOT_QUALIFIED`.

## Frozen qualification sampling contract versions

`datasets/schema/g3-qualification-run-contract.v2.schema.json` freezes the
planned independent-unit count for every expected stratum in each evaluator
branch and requires the per-branch total to equal the sum of those strata. The
roster must exactly match the shared acceptance family. Its minimum counts are
the executable floors needed by the current scorer and the material evaluator's
pilot interval floor: 20 material acquisition units, 2 complex-CSI clusters, 2
propagation clusters, and 20 receiver clusters per stratum. Path equivalence
binds exactly one physical episode to each frozen episode-roster row. These
floors make a contract executable; they are not a powered sample-size design or
evidence that the units are independent. Contract v3 adds a hash-bound joint
precision/power design, and contract v4 adds the raw path-analysis settings
required by input v5. Every contract version remains `NOT_QUALIFIED` until its
evidence and review gates pass.

Create a manifest after generating the input and numerical artifact:

```bash
python -m evaluation.g3_joint_evidence_run_manifest \
  --create-run-id joint-run-001 \
  --input joint-input.json \
  --artifact joint-report.json \
  --output run-manifest.json
```

It prints a JSON report and returns a nonzero exit status for an inconsistent
manifest or failed replay.

This manifest is a run receipt, not a v0.4 G3.03 acquisition package. It does
not crosswalk branch records to signed acquisition-file inventory entries,
authenticate loaded evaluator code, provide external acquisition attestation,
or authorize reviewer roles. G3 remains open, and production D1 generation
remains gated.

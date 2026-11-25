# evaluation.run_manifest

Joint-evidence run manifests, split by dependency layer.

## Modules

| Module | Contents |
| --- | --- |
| `constants.py` | Schema paths, signature namespaces, byte limits and evaluator source rosters |
| `common.py` | Canonical JSON and runtime identity helpers |
| `sources.py` | Evaluator source bundle identity and rooted reads |
| `signatures.py` | Signed runtime inventory, signer authorization and report checks |
| `contracts.py` | Joint evaluator input schema and qualification contract bindings |
| `build.py` | Construction of signed run manifests |
| `verify.py` | Verification of run manifests against recorded inputs and sources |
| `cli.py` | Command-line entry point |

The public names are re-exported from `evaluation/joint_evidence_run_manifest.py`, which keeps the original path named by the evaluator rosters. Its `__main__` guard stays there.

## Invariants

- The source-file check requires the recorded digest to equal the current file digest.
- The rosters in `constants.py` are unchanged; the new modules are outside the signed roster (narrowed binding, see `docs/architecture/module-split-design-notes.md`).

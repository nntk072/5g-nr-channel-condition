# evaluation.acquisition

Acquisition-package verification, split by dependency layer.

## Modules

| Module | Contents |
| --- | --- |
| `constants.py` | Schema paths, size limits and limitation text |
| `noise.py` | Noise RMS summary |
| `canonical.py` | Canonical JSON, timestamp and strict-object helpers |
| `files.py` | Bounded and root-relative file reads |
| `index.py` | Index and pointer helpers for inventories |
| `verify.py` | Verification of acquisition packages |

The public names are re-exported from `evaluation/acquisition_package.py`, which keeps the original path named by the evaluator rosters.

## Invariants

- Behaviour is unchanged by the split; the facade is the import surface.
- The facade path is in the V1, V2 and V4 rosters; the new modules are outside them (narrowed binding, see `docs/architecture/module-split-design-notes.md`).

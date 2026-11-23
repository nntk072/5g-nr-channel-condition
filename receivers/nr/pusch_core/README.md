# receivers.nr.pusch_core

NR PUSCH intra-slot hopping, split by dependency layer.

## Modules

| Module | Contents |
| --- | --- |
| `constants.py` | Carrier-grid limits |
| `types.py` | Dataclass grids for intra-slot hopping |
| `dmrs.py` | DM-RS plans and grids |
| `mapping.py` | Data-symbol mapping and allocation checks |
| `estimation.py` | Channel estimation and demapping |
| `hopping.py` | Composition and dehopping of hops |

The public names are re-exported from `receivers/nr/pusch.py`, which keeps the original path that about 18 tools and retained results name. `NRPUSCHConfig` and `NRChannelEstimate` stay in the facade, because their module path is part of the audited model-input identity.

## Invariants

- Dependencies point down: `constants` and `types` are the base; no cycles.
- The facade is the import surface; the submodules are outside the digests that the audit tools name (narrowed binding, see `docs/architecture/module-split-design-notes.md`).

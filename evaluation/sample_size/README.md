# evaluation.sample_size

Sample-size planning for the joint evidence, split by dependency layer.

## Modules

| Module | Contents |
| --- | --- |
| `constants.py` | Numeric limits and claim-count constants |
| `binomial.py` | Binomial and bounded-mean design helpers |
| `independent_units.py` | Independent acquisition-unit planning |
| `joint_design.py` | Composition and verification of the joint design |
| `bootstrap.py` | Complex-CSI and bootstrap assurance planning |

The public names are re-exported from `evaluation/sample_size_planning.py`, which keeps the original path named by the evaluator rosters.

## Invariants

- Dependencies point down: `constants` and `binomial` are imported by the planners; no cycles.
- Behaviour is unchanged by the split; the facade is the import surface.

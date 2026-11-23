# Research plan topic: architecture

This note holds the architecture decisions from the archived master plan. It
replaces the edition-numbered plan as the place to read them.

**Source:** the archived plan at git tag `docs/master-research-plan-archive`
(file `docs/research/MASTER_RESEARCH_PLAN.md`), sections 1.1, 1.4, 4.6 and
Appendix B. Current gate status is in [MVP_STATUS.md](MVP_STATUS.md), not here.

## Existing components to keep and test

The archived audit listed the components that were reliable starting points:
provenance and atomic manifests, parameter loading, the complex channel dataset,
the Sionna PHY and RT adapters, the physics transforms, and the MLP, RF and CORAL
baselines. Each entry also named its limit. For example, the pilot approximation
is not a 3GPP DMRS model, and the synthetic debug data is for smoke tests only.

## Target layout

The plan set this package layout. Each area owns one concern:

- `integrations/sionna/`: version gate, PHY and RT adapters, and native contracts.
- `propagation/`: ray-path schema, units, label definitions, targets and interventions.
- `receivers/`: the NR PUSCH receiver, DMRS, synchronization, impairments and masks.
- `models/` and `learning/`: real and complex baselines, the path-set decoder, uncertainty, losses and training.
- `forecast/`: dynamic datasets, channel forecasting, blockage and beam decisions.
- `evaluation/`: independent splits, inference contract, statistics, calibration and reporting.
- `validation/`: invariants, oracle tests, native Sionna checks, leakage scans and visual QA.
- `config/`: physics, receiver, dataset, scene, model and experiment configuration.

The current tree differs from this layout in places. The archived plan says the
change is additive, not a rewrite, so existing domain owners are kept.

## Execution environments

Sionna RT 2.2 and Sionna PHY 2.2 need a pinned environment, and the native RT
path needs a compatible Mitsuba and Dr.Jit backend. The plan requires that an
incompatible version fails loudly, with no silent fallback to a legacy RT path.
On hosts where the GPU RT backend is unavailable, CPU LLVM is used for physical
checks, and the physics model is never switched silently.

## Interface boundaries

The architecture separates three roles that must not be mixed:

- **Oracle** (`H_true`, path truth, geometry, profile identity): used for targets and auditing only.
- **Receiver observation** (`H_hat`, masks, known receiver configuration): the only model input.
- **Independent measurement**: the external test, never used for tuning.

A training loop receives two separately scoped objects. `Model.forward` must not
accept the private oracle. The plan states this as an interface contract, and the
leakage scanner enforces it on the data side.

## Provenance and change control

Every artifact carries a schema version, a code commit, package and lock
identity, and the hashes of its inputs. A change to units, conventions, masks,
targets, materials, Sionna flags, pilot grids, split identifiers or metric
aggregation creates a new contract version and new results. Results are never
patched in place. See [PLAN_QUALIFICATION_GATES.md](PLAN_QUALIFICATION_GATES.md).

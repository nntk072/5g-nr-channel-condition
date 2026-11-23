# Research plan topic: data generation

This note records the data program from the archived master plan: the data
tiers, the scene families, the validation of geometry, the link and trajectory
protocol, the storage schema, the label policy and the real-measurement gate.

**Source:** the archived plan at git tag `docs/master-research-plan-archive`
(file `docs/research/MASTER_RESEARCH_PLAN.md`), sections 2.2–2.4 and 5.1–5.8.

## Data tiers

Each tier answers one question and must not be reported as another tier.

| Tier | Data | Used for | Never claim |
| --- | --- | --- | --- |
| D0 | Analytical free-space, two-ray and known-Doppler channels | unit tests and oracle checks | realistic field performance |
| D1 | Sionna PHY TDL/CDL A–E | standardized controls and profile recognition | geometric LOS truth |
| D2 | Scenario models (UMi, UMa, RMa, InH, InF) | bridge from profiles to scenarios | equivalence to detailed RT |
| D3 | Controlled RT canonical scenes | causal mechanisms and sensor resolution | complete physical city fidelity |
| D4 | Independent high-fidelity RT scenes | held-out scene generalization | validation against the world |
| D5 | Permission-cleared NR CSI or I/Q with surveyed truth | external transfer and simulator-bias audit | absolute LOS from subjective signal strength |

The three checked-in city scenes are deterministic controlled fixtures. They are
not enough for external-validity claims, because one procedural rule family and
simple box meshes make held-out seeds easier than an independent scene.

## Label semantics

Profile identity is not geometric LOS. Standardized profiles A–C and D–E are
controlled pretraining sources, and a profile-only oracle can explain a high
score when the profile families differ in LOS state. The plan requires a
profile-only audit and holdout by whole profile family.

Geometric LOS, observable direct power and link coverage are separate labels.
A physically present direct path below the noise floor can be geometric LOS and
not observable at the same time. Both values can be correct, and the plan keeps
that case rather than treating it as a contradiction.

For each target the plan requires an oracle definition, a label scope (per path,
per antenna, per link or per trajectory), a validity mask, a tolerance, a
quality flag and a label version.

## Geometry validation

Every scene version has an immutable manifest with its source, licence,
checksum, coordinate frame, unit system, material assignments and test outputs.
The validation covers mesh orientation, placement outside solid obstacles,
agreement between the direct-segment oracle and the solver on reviewed examples,
deterministic replay, path-delay ordering, the coherent and incoherent power
conventions, and convergence in samples per source, depth and mesh precision.
Solver failures and unresolved links are reported separately.

## Link and trajectory protocol

Static samples are drawn before the label distribution is inspected. Acceptance
statistics are recorded in full, and a balanced diagnostic subset is kept apart
from the representative distribution. Rejection sampling must not hide the LOS
ratio.

Dynamic data uses physically meaningful trajectories with a declared cadence,
velocity and turn limits, and a geometry hash per time stamp. A blocker
counterfactual re-traces the whole scene with and without the blocker. Deleting
one path coefficient is a mathematical ablation, not a physical intervention.

Groups are assigned at scene or asset-family, block, trajectory and measurement
session levels, with a spatial buffer across split boundaries. The plan requires
a nearest-neighbour audit of the splits.

## Storage schema

The archived plan proposes one sample record with four parts: the receiver
observation, the privileged simulation fields, the target values and the
provenance. The schema is a design, and the plan says it was not yet serialized
in that form. Bulky arrays are stored by checksum, and the schema forbids implicit
flattening of antenna and time axes.

## Real measurements

The real-measurement gate needs a written permission, hardware and clock
documentation, timing alignment, surveyed visibility labels that do not depend on
the CSI, and a held-out site that is never used for tuning. A recording session
must never be split between training and test. If real data is not available, the
claim is limited to simulation.

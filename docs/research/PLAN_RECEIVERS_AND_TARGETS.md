# Research plan topic: receivers, targets and methods

This note records the observation model, the receiver assumptions, the target
inventory and the methodological program from the archived master plan.

**Source:** the archived plan at git tag `docs/master-research-plan-archive`
(file `docs/research/MASTER_RESEARCH_PLAN.md`), sections 0, 1.2, 2.1–2.4, 3, 3.1
and 6.

## Headline claim and novelty rule

The plan's single claim is conditional, and it must be pre-registered: under
limited bandwidth, SNR, antenna and domain conditions, an identifiability-aware
physics-consistency method achieves better joint estimation quality and
uncertainty reliability on unseen environments than classical estimators and
parameter- and compute-matched MLP, CNN and CVNN baselines, without privileged
labels at inference.

Novelty is not assumed from the name or the general architecture. A candidate
mechanism must have a testable property beyond known complex convolutions,
equivariance, CORAL, ordinary multitask heads or generic path-token
transformers. Its same-backbone model with that mechanism switched off is the
comparison.

## Observation model

The idealized channel is a sum of finite rays, one term per path, each with an
amplitude, an angle-dependent array response and a delay phase. A pilot estimate
is the output of a receiver process: the transmit and receive observation
process, the NR resource grid, and the noise, synchronization and hardware
terms. The learning model sees only the estimate, the masks, the documented
receiver uncertainty and the calibration parameters a real receiver would know.

Timing, phase and array conventions are gauge freedoms. A common phase rotation
and a common delay shift must leave the LOS decision and the delay spread
invariant where the physics says so. Absolute time of arrival is not inferable
unless a synchronized reference exists.

## Resolution and identifiability

Time resolution follows the occupied bandwidth, but it is not a hard lower bound
for every estimator. Uniform frequency bins make the delay phase periodic, so
paths beyond the unambiguous window must be labelled separately. Close paths may
not be identifiable, and the plan asks for an uncertainty or a merged resolvable
cluster in that case. A solver that returns no ray at a finite budget is a
censoring case, not proof that the path does not exist.

Angles need an illuminating array and calibrated geometry. Doppler needs several
samples in time. A single snapshot cannot support a Doppler or forecasting claim.

## Target inventory

The plan lists 22 targets, each with an oracle, an observability condition and
main metrics. The first paper's required group is: LOS and direct detectability
(T01–T03), delay spread and first-path delay with the PDP (T04, T06, T11),
masked complex reconstruction (T16) and calibration (T21). The plan adds path
count, the direct-to-indirect ratio and the strongest-path delay when they are
reliable. Angles, angular spread, Doppler, path-wise gains and absolute ToA
require expensive array or measurement validation and may become a later paper.

A reconstruction loss uses held-out observable target samples, with the measured
noise model and no train-test leakage. A noise-free reference is used only in
simulation studies.

## Receiver realism

The receiver audit records bandwidth, the number of used resource elements, the
physical resource blocks, the FFT size, guard subcarriers, the cyclic prefix and
the real pilot positions. A 2.88 MHz nominal allocation and a synthetic 3.84 MHz
grid are different experiments, and the plan forbids treating them as the same.

Relative-SNR noise and thermal-noise link budgets are separate conditions, and
each must be labelled. Only a validated receiver can feed the receiver-realistic
claims.

## Methodological program

The plan's order is: correctness gates, physically credible data, matched
baselines, one new mechanism, independent transfer, dynamic and forecasting
extension, downstream impact, and then the paper. The baseline must be a robust
parameter-matched MLP and the classical estimators. A CNN contribution must be
justified by a matched-input, matched-budget module ablation. If the MLP matches
the proposed model, the plan says to use the MLP as the encoder and to contribute
the physical decoder or loss instead.

The primary venue choice is wireless communications first, and a general machine
learning venue only if the learning principle generalizes beyond the application.

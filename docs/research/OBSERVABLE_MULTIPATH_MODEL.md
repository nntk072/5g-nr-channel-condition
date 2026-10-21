# Analytical two-path decoder: implementation contract

`models/observable_multipath.py` provides an initial differentiable component
for the M2/E11 research gate. It fixes the common-delay gauge and models the
relative-delay observation as

\[
H(f_k) = \alpha_0 + \alpha_1 e^{-j 2\pi f_k \Delta\tau} + n_k.
\]

`render_two_path_cfr()` is the explicit Torch-forward map and accepts either a
shared frequency vector or a per-example frequency grid. The MLP factory
`build_two_path_uncertainty_decoder()` accepts complex CFR pilot observations,
their mask, the configured tone frequencies, and per-example complex noise
variance. It retains log occupied-frequency span in the features; normalized
frequency coordinates alone would erase the absolute bandwidth needed to
estimate delay in seconds across bandwidths. It returns two complex gains, an ordered relative delay bounded by
the configured maximum, and a positive heteroscedastic delay scale. Gains are
reported in the CFR's original units; the model standardizes inputs by the
known noise standard deviation. `complex_pilot_gaussian_nll()` scores only
observed pilots, while `heteroscedastic_delay_nll()` supplies a Gaussian
relative-delay loss.

`experiments/observable_multipath.py` implements grouped analytical data
generation, group-disjoint train/validation/calibration/test assignment,
training with validation checkpoint selection, and a variable-projection
profile-likelihood baseline with local curvature scale. It also provides a
local Fisher-information delay CRLB after projecting out the unknown complex
path gains. The matched identifiability-on arm adds a predeclared penalty when
the decoder's delay scale falls below that local bound; the off arm has the
same architecture, initialization seed, data, and training budget. This is one
testable calibration regularizer, not a complete equivalence-aware posterior.
The runner
`scripts/run_observable_multipath_experiment.py` freezes neural and classical
predictions for both ablation arms and the classical estimator, then sends them
through the grouped uncertainty evaluator. The
runner publishes a hash-indexed output directory atomically and refuses to
overwrite an existing run. It also reports paired group-bootstrap deltas for
the on/off ablation and classical comparisons, using held-out test groups as
the resampling unit. Decoder checkpoints use a pickle-free NPZ state dictionary;
`load_two_path_uncertainty_decoder()` restores the declared architecture from
the recorded model configuration. Test-time model inputs remain the receiver-visible
pilot observations, masks, frequencies, and noise estimate. The common absolute
arrival time is deliberately not estimated because it is confounded with an
unknown complex path gain under this observation model.

This is a fixed-two-path analytical baseline and software foundation, not a
complete OMSI method or evidence of identifiability-aware improvement. It has
no path-existence head, diffuse residual model, array manifold, clock model,
or measured-channel validation. The first training experiment samples
synthetic groups from one condition mixture; its held-out metrics are pilot
diagnostics, not independent-scene transfer or a scientific advantage claim.
The M2 gate still
requires mathematical equivalence-class decisions, a same-backbone
identifiability-on/off comparison, matched classical and neural baselines,
calibration on independent groups, and evaluation on qualified RT scenes and
eventually measured NR data. The decoder must not be used to claim physical path
count when the pilot observation cannot resolve the components.

Experiment schema v2 also reports a set-valued profile-likelihood comparator.
For each bounded relative-delay candidate, it eliminates the unknown complex
gains and evaluates the complex-Gaussian likelihood-ratio deviance. A threshold
is calibrated from the maximum true-delay deviance within each independent
calibration group. Test outputs retain disconnected sublevel-set intervals
instead of forcing ambiguous delay hypotheses into one interval. The threshold
has the stated group-conformal interpretation only under exchangeable groups
and the fixed-two-path, known-noise model; the finite grid approximates the
continuous set, and this analytical result does not transfer to RT or measured
channels without a separate validation.

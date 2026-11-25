# models

**Purpose.** Model definitions for the research baselines: classical multipath
fitting, capacity envelopes, real- and complex-valued convolutional networks,
and a differentiable observable multipath model.

**Public API.**

- `models.classical` — estimator construction for the classical baselines.
- `models.multipath` — sparse multipath fitting with data-driven model-order
  selection.
- `models.capacity` — explicit neural-baseline capacity envelopes.
- `models.real_cnn`, `models.complex_cnn` — CNN baselines for complex CFR
  observations (real/imaginary input and phase-equivariant complex input).
- `models.observable_multipath` — two-path observation model with an
  uncertainty-aware decoder.
- `models.network` — neural model definitions; framework imports are delayed
  until a model is constructed.

**Invariants.**

- Model definitions contain no training loop. Training lives in `learning`.
- Geometric line-of-sight labels and path-oracle truth are never inputs to a
  model here. Only receiver-visible observations are.
- Deep-learning frameworks are imported lazily, so the package imports on a
  machine without them.

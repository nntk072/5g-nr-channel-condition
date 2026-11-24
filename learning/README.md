# learning

**Purpose.** Training loops and supervised fitting entry points for the
baselines defined in `models`.

**Public API.**

- `learning.supervised` — supervised fitting entry points for the standard
  baselines.
- `learning.real_cnn` — training loop for the real/imaginary CNN.
- `learning.complex_cnn` — training loop for the phase-equivariant complex CNN.
- `learning.coral` — CORAL representation alignment and portable checkpoint
  loading.

**Invariants.**

- Splits are taken from the dataset's frozen split registry. Training code does
  not create its own split.
- Checkpoints are saved in a portable format and loaded with an explicit
  architecture check.
- Training requires the optional PyTorch dependency; tests that need it skip
  cleanly when it is absent.

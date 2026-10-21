# 2026 Sionna Research Upgrade

The original Tampere 2025 thesis and notebooks are preserved under
`notebooks/legacy_2025/` and `archive/legacy_2025/`. The reproducible research
framework has explicit domain packages and tests under `tests/`. This branch targets *genuine physical LOS/NLOS detection*
under cross-domain distribution shift, not reclassification of TDL/CDL labels.

**Start:** `docs/research/QUICKSTART.md`

**Repository architecture:** `docs/architecture/README.md`

**Paper objective & experimental roadmap:** `docs/research/RESEARCH_PROGRAM.md`

**Legacy baseline audit:** `docs/research/BASELINE_AUDIT.md`

**Extra modules:** `loslab/geometry.py` generates independent deterministic urban scenes;
`loslab/pilots.py` simulates noisy non-standardized comb pilot LS estimates.
All model checkpoints are stored with evaluation predictions.

**Quick self-test:** `python -m pytest -q`

**Native Sionna 2.2 status:** Not validated on the construction host; the
version-gated PHY/RT code needs native execution in Linux/WSL2.

**Research status:** framework/experiment design, not a claimed publication
result. Artificial debug experiments MUST NOT be cited as accuracy evidence.

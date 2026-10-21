# 5G Channel LOS/NLOS Thesis

Research code for machine-learning-based channel classification, with current
work focused on physically grounded LOS/NLOS detection under domain shift. The
research framework distinguishes geometric LOS labels from standardized
TDL/CDL profile proxy labels and marks synthetic smoke data as debug-only.

## Start here

- [Research quickstart](docs/research/QUICKSTART.md)
- [Research program and evidence limits](docs/research/RESEARCH_PROGRAM.md)
- [Repository architecture](docs/architecture/README.md)
- [Full phased refactor plan](docs/architecture/REFACTOR_PLAN.md)
- [2025 thesis PDF](docs/thesis/NguyenLong.pdf)

## Install and run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m loslab doctor
```

Sionna PHY and ray tracing are optional integrations. Install the `sionna`
extra only in an environment with compatible PyTorch, CUDA, Mitsuba, and Dr.Jit;
see the quickstart before running native simulations.

## Repository map

```text
artifacts/       run identity, manifests, and provenance
configuration/  config loading and canonical hashes
config/          versioned experiment and runtime declarations
datasets/        channel dataset schema and data generators
propagation/     propagation physics and deterministic scene geometry
receivers/       channel observation and pilot estimation
models/          classifier implementations
learning/        training objectives and adaptation algorithms
evaluation/      metrics, group-aware splits, and statistics
experiments/     experiment orchestration
integrations/    optional Sionna adapters
data/ scenes/ cache/ outputs/
notebooks/       active and historical notebooks
archive/         preserved superseded 2025 implementation
tests/           tests organized around active packages
```

The Sionna source checkout used as an API reference is kept separately at
`/home/nguyenl37/sionna`; it is not vendored into this project or imported by
the runtime.

Author: Long Nguyen

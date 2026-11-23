# Quickstart

This page shows how to set up the environment and run the repository gates. The
gates are the same commands CI runs; see [Continuous integration](../architecture/ci.md).

## Set up the environment

Python 3.12 is the supported version. From the repository root:

```bash
uv sync --locked --extra dev
```

The neural and boosted-baseline gate needs the boosting extra and CPU-only PyTorch:

```bash
uv sync --locked --extra dev --extra boosting
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Sionna ray tracing is an optional research dependency. Its tests are opt-in and
need `CUDA_VISIBLE_DEVICES=-1` and a Dr.Jit LLVM library (`DRJIT_LIBLLVM_PATH`).

## Run the gates

```bash
./tools/check lock
./tools/check lint
./tools/check typecheck
./tools/check test
./tools/check neural
./tools/check build
./tools/check agent
./tools/check docs
```

## Build this site

```bash
uv sync --locked --extra dev --extra docs
./tools/check docs
```

The HTML output is written to `docs/_build/html`.

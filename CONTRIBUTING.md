# Contributing

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional extras: `sionna` (PHY), `sionna-rt` (ray tracing), `boosting`, `docs`.

## Gates

Every check is a named gate in `tools/check`. CI runs the same gates.

| Gate | What it checks |
| --- | --- |
| `lock` | `uv.lock` matches `pyproject.toml` |
| `lint` | ruff lint and formatting for the package and test trees |
| `typecheck` | mypy over the packages |
| `architecture` | package layering, import cycles, package READMEs, package lists |
| `test` | core test suite without coverage |
| `neural` | full suite with coverage (threshold in `pyproject.toml`) |
| `build` | wheel and sdist build, and an import smoke test of every package |
| `docs` | Sphinx build with warnings as errors |
| `fast` | lint, typecheck and tests on the files you changed |

Run a gate with `./tools/check <gate>`. Before opening a change, run `lock`,
`lint`, `typecheck`, `architecture` and `test`.

## Rules for changes

- Keep one concept per module. Put a new package in `config/packages.toml` and
  give it a `README.md` that states its purpose, public API and invariants.
- Do not introduce import cycles. The architecture gate reports them.
- Public functions should have a docstring (not yet enforced by the lint gate; see
  the documentation plan). Modules that build or check scientific artifacts document
  their output schema and the version it carries.
- Name things by what they compute or check. Do not put phase, status or
  version words in file, module or function names. Schema versions are the only
  place a version appears, as a `vN` suffix on the schema file.
- Native Sionna tests are marked `native_sionna` and need the Sionna environment;
  the default gates skip them. Say in the change which native tests you ran.
- Do not commit generated outputs, caches, data or local developer notes.
  `dev-docs/` and `data/` are git-ignored.

## Tests

Tests live in `tests/`, mirroring the package layout. A test should exercise
behaviour that a product or a documented workflow depends on. Tests that only
recompute a stored record are not accepted.

## Commit messages

Use an imperative subject line of at most 72 characters and a body that says why
the change is needed.

# 5G NR Channel Condition

Software for generating and evaluating NR channel-condition datasets and
receiver observations. It covers:

- standardized TDL and CDL channel generation and NR PUSCH/DM-RS observations
  through an optional Sionna PHY integration;
- ray-traced channel datasets through an optional Sionna RT integration;
- immutable, content-addressed storage of channel records and episode bundles,
  with scene-grouped split policies and leakage audits;
- evaluation contracts for metrics, paired statistics, and acceptance, each
  with an explicit scope.

The software distinguishes geometric line-of-sight labels, standardized
profile labels, and synthetic debug data. Results carry their configuration and
source hashes. A passing check is an engineering result. It does not qualify a
scene, a receiver, or a standard.

## Documentation

- [Documentation index](docs/README.md) lists every guide, architecture note,
  and reference page.
- [Getting started](docs/guide/getting-started.md) explains setup and the
  quality gates.
- [Architecture overview](docs/architecture/overview.md) describes packages and
  data flow.
- [Command reference](docs/reference/cli.md) lists the installed commands.
- [Schema index](docs/reference/schemas.md) lists the versioned JSON schemas.

## Quick start

```bash
uv sync --locked --extra dev
./tools/check lint
./tools/check test
loslab-runtime-inventory --help
```

The native PHY and RT integrations need optional extras and a suitable host:

```bash
uv sync --locked --extra dev --extra sionna        # Sionna PHY
uv sync --locked --extra dev --extra sionna-rt     # Sionna RT (Mitsuba, Dr.Jit)
```

Native tests skip cleanly when their dependencies are absent. Their results are
evidence only for the host on which they run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, the quality
gates, and the coverage policy. Report security issues as described in
[SECURITY.md](SECURITY.md). Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

No license has been chosen for this repository yet. The license decision is an
open owner decision, recorded in the reading plan. No license is granted until
that decision is made.

## Citation

See [CITATION.cff](CITATION.cff).

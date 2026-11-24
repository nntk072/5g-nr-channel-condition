# Coverage policy

The CI gate requires **100 % branch coverage** of the product packages
(`[tool.coverage.run] source` in `pyproject.toml`).

Exclusions are the only way a line or branch may be left out of the gate. Each
one is listed below with its reason. `tools/coverage_policy.py` runs in the
architecture gate and fails when a configured exclusion has no row here, or when
a row no longer matches the configuration.

Each row is a single table line: the first column is the exact configured entry
in code formatting, the second column is the reason.

## Exclusions

| Entry | Reason |
| --- | --- |
| `if TYPE_CHECKING:` | Import-time branch for type checkers only; it never runs. |
| `if __name__ == .__main__.:` | Module entry-point guard. The behaviour is tested through the callable `main()`. |
| `integrations/sionna/rt_trajectory.py` | Sionna RT trajectory generation drives Mitsuba/Dr.Jit scene transport. Neither package is installed in CI (only the `sionna` PHY extra is), so these statements cannot execute there. Their native RT evidence is retained separately. Pure helpers in this module are not measured until they are split into an RT-free module. |
| `integrations/sionna/rt_network.py` | Network-level Sionna RT ray tracing (Mitsuba/Dr.Jit transport). Requires the `sionna-rt` runtime that CI does not install, so the statements cannot execute in the gate. Pure geometry helpers are not measured until split out. |
| `integrations/sionna/adapter/raytrace.py` | Sionna RT adapter that calls the ray tracer and its compiled transport. The runtime is not installed in CI, so the adapter cannot execute there. |
| `integrations/sionna/adapter/generate.py` | Standardized channel generation through the Sionna RT ray tracer. Requires the `sionna-rt` runtime absent from CI; the RT-free record assembly is covered through the standardized-run tests. |
| `integrations/sionna/path_lineage.py` | Observer that patches Sionna RT internals to record path lineage. Patching requires the `sionna-rt` runtime; the observer's RT-free methods are covered by fake-based tests, and the attach path needs native RT evidence. |

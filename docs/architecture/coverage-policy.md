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
| `if not hasattr\(os, "O_NOFOLLOW"\)` | Platform refusal. Raised only where the operating system lacks the `O_NOFOLLOW` open flag. CI and development run on Linux, which always defines it, so the branch cannot execute in CI. |
| `if not hasattr\(os, "O_DIRECTORY"\) or os\.open` | Platform refusal for systems without directory-relative opens (`os.open` and `os.supports_dir_fd`). Linux always provides them, so the branch cannot execute in CI. |
| `if os\.name != "posix"` | Early return on non-POSIX systems, where directory fsync and no-follow opens are unavailable. CI runs only on Linux, so the early return cannot execute there. |
| `if device\.startswith\("cuda"\) and not torch\.cuda\.is_available` | CUDA-only error path: raises when a CUDA device is requested but unavailable. The CI neural job runs on a CPU-only runner, so this path cannot execute there. The CPU path of the same function is measured. |
| `if not torch\.cuda\.is_available\(\)` | CUDA-only error path: refuses to run a native workflow that requires a GPU. The CI neural job has no GPU, so this path cannot execute there. |
| `if not torch_cuda_available\(\)` | CUDA-only error path for a workflow that requires CUDA. The CI neural job has no GPU, so this path cannot execute there. |
| `if temp_path is not None:` | Temporary-lock cleanup guard: normal return requires a created temporary path; when creation fails before that assignment, the original exception propagates and the function cannot reach its return. The guard only controls cleanup during exception unwinding. |
| `integrations/sionna/rt_trajectory.py` | Sionna RT trajectory generation drives Mitsuba/Dr.Jit scene transport. Neither package is installed in CI (only the `sionna` PHY extra is), so these statements cannot execute there. Their native RT evidence is retained separately. Pure helpers in this module are not measured until they are split into an RT-free module. |
| `integrations/sionna/rt_network.py` | Network-level Sionna RT ray tracing (Mitsuba/Dr.Jit transport). Requires the `sionna-rt` runtime that CI does not install, so the statements cannot execute in the gate. Pure geometry helpers are not measured until split out. |
| `integrations/sionna/adapter/raytrace.py` | Sionna RT adapter that calls the ray tracer and its compiled transport. The runtime is not installed in CI, so the adapter cannot execute there. |
| `integrations/sionna/adapter/generate.py` | Standardized channel generation through the Sionna RT ray tracer. Requires the `sionna-rt` runtime absent from CI; the RT-free record assembly is covered through the standardized-run tests. |
| `integrations/sionna/path_lineage.py` | Observer that patches Sionna RT internals to record path lineage. Patching requires the `sionna-rt` runtime; the observer's RT-free methods are covered by fake-based tests, and the attach path needs native RT evidence. |

# Sionna RT run identity

Every generated RT dataset or trajectory should have one identity derived from
the full local scene-asset closure, physical configuration, solver options, RNG
seed, active source-code digest, Sionna/Mitsuba/Dr.Jit versions, and the
successfully probed Mitsuba backend. The identity is stored in the generation
report (or trajectory sidecar) as `rt_run_identity.cache_key_sha256`.

`auto` backend selection probes a real deterministic direct-path solve in an
isolated process. It selects `cuda_optix` only when the native trace succeeds;
otherwise it tries `llvm_cpu`. The selected variant and native versions are
part of the run identity, so the two execution paths cannot collide. An
explicitly requested backend still fails with diagnostics rather than silently
changing execution mode.

If any scene resource cannot be resolved to a local file, the manifest is
incomplete and the identity has `complete: false` and a null cache key. Such a
run can still be executed and audited, but callers must not reuse its outputs as
content-addressed cache entries. The key is an identity contract; it does not
implement a cache store or establish numerical equivalence across hardware.

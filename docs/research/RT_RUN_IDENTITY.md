# Sionna RT run identity

Every generated RT dataset or trajectory has an identity derived from the full
local scene-asset closure, physical configuration, solver options, RNG seed,
active source-code digest, Sionna/Mitsuba/Dr.Jit versions, the successfully
probed Mitsuba backend, and runtime details (OS, machine architecture, Python
implementation, device-selection variables, and visible NVIDIA device/driver
fingerprint when `nvidia-smi` is available). The identity is stored in the
generation report (or trajectory sidecar) as `rt_run_identity.cache_key_sha256`.

`auto` backend selection probes a real deterministic direct-path solve in an
isolated process. It selects `cuda_optix` only when the native trace succeeds;
otherwise it tries `llvm_cpu`. The selected variant and native versions are
part of the run identity, so the two execution paths cannot collide. An
explicitly requested backend still fails with diagnostics rather than silently
changing execution mode.

If any scene resource cannot be resolved to a local file, the manifest is
incomplete and the identity has `complete: false` and a null cache key. Such a
run can still be executed and audited, but callers must not reuse its outputs as
content-addressed cache entries. Static RT datasets can opt into the immutable
cache with `loslab rt-data --cache-dir PATH`. The cache stores an NPZ payload
and JSON sidecar under the identity key, verifies payload/report hashes and
dataset shape on every read, publishes files atomically, and rejects different
output bytes for an already-published identity. A missing or partial pair is
treated as an error rather than a cache miss. A cache hit still probes and
activates the selected runtime backend so identity resolution remains tied to
a backend that works in the current environment. The same runtime fingerprint
is passed from cache-key construction through generation to avoid recomputing
device state mid-run. Trajectory generation currently records identities but
does not use this static-dataset cache. Cache identity prevents incompatible
runtime reuse; it does not establish numerical equivalence across hardware.

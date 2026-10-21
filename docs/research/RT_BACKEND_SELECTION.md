# Sionna RT backend selection

RT generation accepts `mitsuba_variant="auto"`, `"cuda"`, `"llvm"`, or an
exact Mitsuba variant name. Automatic mode tries
`cuda_ad_mono_polarized` first, then `llvm_ad_mono_polarized`.

An installed CUDA variant does not prove that an RT run can use OptiX. The
selector runs a deterministic free-space direct-path trace in a child Python
process for each candidate. This checks the Mitsuba variant, Dr.Jit runtime,
Sionna RT import, OptiX-dependent solver initialization, and one returned path.
The probe is isolated because a failed GPU runtime initialization can leave
global Dr.Jit/Mitsuba state unsuitable for switching backends in the parent.
When CUDA probing fails, `auto` records the reason and activates LLVM. An
explicit `cuda` request fails with the probe error instead of silently falling
back. Scene records and trajectory manifests store the requested/resolved
variant, backend label, installed variants, probe result, and fallback reason.

The basic implementation is:

```python
from integrations.sionna import select_rt_backend, activate_rt_backend

selection = select_rt_backend("auto")
activate_rt_backend(selection)  # before importing sionna.rt or loading a scene
print(selection.as_dict())
```

Run the same preflight from a prepared environment with:

```bash
uv run --no-sync --locked python scripts/check_rt_backend.py --backend auto
```

Sionna RT currently requires the polarized monochromatic variants used by this
repository. A generic `cuda_ad_rgb` Mitsuba variant is therefore not considered
a substitute. The probe tests one simple direct path, not scene-specific
convergence, physical accuracy, memory headroom, or production ray budget.
Re-run the normal numerical and scene QA suite after selecting a backend for a
new host. Historical data/configurations that explicitly record LLVM continue
to use LLVM for exact replay; newly constructed default `RayScene` and RT
trajectory configurations use `auto`.

The solver-convergence, ray-budget replay, and path-QA runners use the same
selector before importing `sionna.rt`. New convergence runs default to `auto`;
replay/audit tools request the exact variant recorded with the dataset and fail
with probe diagnostics if that backend cannot run. Their reports include the
resolved variant and probe/fallback provenance.
The selection record also captures the installed Sionna RT, Mitsuba, and
Dr.Jit distribution versions so a backend result can be interpreted against
the native stack that produced it.

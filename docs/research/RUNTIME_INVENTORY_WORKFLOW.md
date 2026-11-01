# Runtime identity inventory

`loslab-runtime-inventory` emits a local, hash-bearing snapshot of the project
lock files, Git revision and tracked-worktree state, installed numerical/Sionna
distributions, Sionna TR 38.901 model-table files, and (optionally) CUDA device
metadata. It uses installed distribution `RECORD`, `WHEEL`, and `direct_url`
bytes as package-install identity and hashes each discovered model-table file.
Each distribution marks its identity `COMPLETE` only when an installed
`RECORD` manifest is available; otherwise it is explicitly `INCOMPLETE`.

The default path reads package metadata only; it does not import Sionna, Torch,
Mitsuba, or initialize a GPU. Enable the opt-in accelerator query with
`--probe-accelerator`; it imports Torch and reads CUDA device properties but
does not allocate tensors or launch a channel simulation.

Add `--probe-apis` to import each pinned Sionna PHY module and check the
repository-used exports. Add `--probe-rt-apis` as well to check the Sionna RT,
Mitsuba, and Dr.Jit APIs; RT imports are never attempted unless this second flag
is present. API probes construct no scene, channel, or solver. The report lists
distribution version mismatches and missing/import-failing symbols explicitly.
The API rows are a compatibility preflight for the symbols currently used by
the repository, not an ABI guarantee or proof that every backend operation
works on the selected hardware.

On Linux, add `--probe-loaded-native-libraries` to emit runtime inventory v3.
After any requested accelerator/API imports, it reads `/proc/self/maps`,
deduplicates mapped ELF shared-object paths, and hashes the exact file bytes
under 1-GiB-per-file, 8-GiB-total, and 16-MiB-maps ceilings. The receipt lists
hashed paths and bytes plus any paths it could not hash. `COMPLETE` means the
enumerated mapped `.so` files were hashed; `INCOMPLETE` and `UNAVAILABLE` are
explicit non-success states and must not be treated as complete identity.
The optional capture can read multiple gigabytes and is intentionally separate
from the default metadata-only inventory. It remains self-reported file
identity, not loaded-memory attestation or trust evidence.

Run from the repository root:

```sh
.venv/bin/python scripts/report_runtime_inventory.py --output runtime-inventory.json
.venv/bin/python scripts/report_runtime_inventory.py --probe-accelerator --output runtime-inventory-cuda.json
.venv/bin/python scripts/report_runtime_inventory.py --probe-apis --output runtime-inventory-phy.json
.venv/bin/python scripts/report_runtime_inventory.py --probe-apis --probe-rt-apis --output runtime-inventory-rt.json
.venv/bin/python scripts/report_runtime_inventory.py --probe-accelerator --probe-apis --probe-loaded-native-libraries --output runtime-inventory-native-v3.json
```

## V100 CUDA environment

On the Halti V100 host, the installed NVIDIA driver is 575.57.08 and exposes
CUDA 12.9. The PyPI PyTorch 2.14.1 wheel selected by the current lock is the
CUDA 13.0 build and fails CUDA initialization on that driver. After syncing the
locked `sionna` extra, install the official CUDA 12.6 wheel in `.venv`:

```sh
uv sync --locked --extra sionna --extra dev
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu126 \
  --reinstall 'torch==2.14.1+cu126'
```

Repeat the second command after a locked sync, because the lock records the
PyPI distribution and does not select a CUDA-specific wheel index. Confirm the
runtime and both devices before any receiver run:

```sh
.venv/bin/python - <<'PY'
import torch

assert torch.__version__ == "2.14.1+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.is_available()
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
assert torch.cuda.get_device_capability(0) == (7, 0)
value = torch.ones((16, 16), device="cuda:0")
assert (value @ value).sum().item() == 4096
PY
```

The extra wheel selection is a host-specific runtime choice; record its exact
distribution identity with `--probe-accelerator` and, when needed, the opt-in
native-library inventory. A successful CUDA kernel smoke establishes device
execution only. Sionna receiver initialization and the bounded receiver
protocol must still be verified separately. Keep `sionna-rt` out of this
PHY-only environment: the installed RT package makes Sionna's top-level
initializer import Mitsuba/Dr.Jit even for `sionna.phy.nr`, and Dr.Jit 1.5
skips its CUDA backend on the server's Volta 7.0 devices. Use a separate
RT-enabled environment for RT-only work. PyTorch's published version/wheel
matrix is at <https://pytorch.org/get-started/previous-versions/>.

Validate v2 receipts against `datasets/schema/runtime-inventory.v2.schema.json`
and opt-in native-library v3 receipts against
`datasets/schema/runtime-inventory.v3.schema.json` (the v1 schema remains for
historical reports). Do not overwrite a retained
receipt; choose a new output name for each environment snapshot. The report is
self-reported integrity metadata only. It is not an attestation, does not prove
reviewer or key custody, and always declares `NOT_QUALIFIED`.

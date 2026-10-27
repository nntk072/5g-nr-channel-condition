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

Validate v2 receipts against `datasets/schema/runtime-inventory.v2.schema.json`
and opt-in native-library v3 receipts against
`datasets/schema/runtime-inventory.v3.schema.json` (the v1 schema remains for
historical reports). Do not overwrite a retained
receipt; choose a new output name for each environment snapshot. The report is
self-reported integrity metadata only. It is not an attestation, does not prove
reviewer or key custody, and always declares `NOT_QUALIFIED`.

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

Run from the repository root:

```sh
.venv/bin/python scripts/report_runtime_inventory.py --output runtime-inventory.json
.venv/bin/python scripts/report_runtime_inventory.py --probe-accelerator --output runtime-inventory-cuda.json
```

Validate the emitted receipt against
`datasets/schema/runtime-inventory.v1.schema.json`. Do not overwrite a retained
receipt; choose a new output name for each environment snapshot. The report is
self-reported integrity metadata only. It is not an attestation, does not prove
reviewer or key custody, and always declares `NOT_QUALIFIED`.

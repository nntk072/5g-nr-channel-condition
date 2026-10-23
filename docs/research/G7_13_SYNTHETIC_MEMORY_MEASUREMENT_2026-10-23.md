# G7.13 synthetic memory measurement

**Status:** bounded synthetic harness evidence only. This is not a production
qualification, full-corpus measurement, or process RSS guarantee.

## Reproduce

From the repository root, run:

```bash
uv run --no-sync --locked python -m pytest --no-cov -s -q \
  tests/test_datasets/test_nr_store_memory_scaling.py
```

The test uses one deterministic compressed NPZ payload containing a zero-valued
`complex64` array of shape `(64, 64)`. It then creates separate synthetic manifest
inventories of 0, 8, 128, and 8,192 records. The loader stages manifest metadata
in a temporary SQLite database, sorts by sample ID there, and limits the SQLite
page cache to 256 KiB. The scan is measured with `tracemalloc`; payload-size
estimation and record decoding are stubbed during this part so the observed
growth represents Python allocations for iteration and one-record validation,
not channel arrays or SQLite's native allocations.

The emitted JSON reports four distinct payload quantities: compressed archive
bytes; uncompressed NPZ member bytes (including NPY headers); the loader's
modeled two compressed buffers; and two uncompressed-member equivalents. The
last two values are conservative arithmetic estimates, not direct allocation
measurements, and the latter does not assert that the reader retains two decoded
copies of each array. The JSON separately reports peak traced Python allocations
for each manifest count. Its exact byte values depend on the Python and NumPy
runtime and are evidence for this synthetic fixture only.

## Interpretation and limits

The reader now uses an external, disk-backed sort instead of retaining a Python
list of all manifest metadata. In the recorded run, peak traced Python allocation
was 16,759 bytes for 8 manifests and 54,318 bytes for 8,192 manifests, a 37,559-byte
increase. These values are runtime-specific and omit SQLite native page-cache
memory, filesystem cache, and process RSS. Temporary-database I/O and disk-space
cost scale with the manifest inventory; this fixture does not extrapolate runtime,
disk use, or memory to a real corpus.

The compressed and uncompressed-member quantities follow the estimate in
`datasets.nr_store._nr_record_memory_estimate`; Python object overhead, allocator
behavior, filesystem cache, native-library allocations, concurrent workers, and
whole-process RSS are outside that estimate. No 100-GB corpus was created or
loaded. G7.13 remains partial until representative corpus-scale behavior is
measured in the intended environment; this synthetic test does not establish a
corpus RSS cap.

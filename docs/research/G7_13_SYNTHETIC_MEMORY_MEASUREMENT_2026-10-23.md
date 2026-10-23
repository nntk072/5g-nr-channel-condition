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
inventories of 0, 8, 32, and 128 records. The manifest scan is measured with
`tracemalloc`; payload-size estimation and record decoding are stubbed during this
part so that the observed growth represents Python allocations for path, parsed
metadata, and inventory data instead of channel arrays.

The emitted JSON reports four distinct payload quantities: compressed archive
bytes; uncompressed NPZ member bytes (including NPY headers); the loader's
modeled two compressed buffers; and two uncompressed-member equivalents. The
last two values are conservative arithmetic estimates, not direct allocation
measurements, and the latter does not assert that the reader retains two decoded
copies of each array. The JSON separately reports peak traced Python allocations
for each manifest count. Its exact byte values depend on the Python and NumPy
runtime and are evidence for this synthetic fixture only.

## Interpretation and limits

The existing reader sorts a corpus-wide manifest inventory before it yields
sample-ID-ordered shards. Therefore inventory memory grows with manifest count,
even though payload arrays are loaded a shard at a time. This test exercises that
shape at a deliberately small scale and checks that the traced peak rises between
8 and 128 manifests. It does not extrapolate the measured slope to a real corpus.

The compressed and uncompressed-member quantities follow the estimate in
`datasets.nr_store._nr_record_memory_estimate`; Python object overhead, allocator
behavior, filesystem cache, native-library allocations, concurrent workers, and
whole-process RSS are outside that estimate. No 100-GB corpus was created or
loaded. G7.13 remains partial until representative corpus-scale behavior is
measured in the intended environment; this synthetic test does not establish a
corpus RSS cap.

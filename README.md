# 5G NR Channel Condition Research

Research framework for machine-learning-based 5G NR channel-condition
analysis. The current study begins with physically grounded LOS/NLOS detection
under domain shift, while the project is structured to grow toward broader
channel-condition characterization and prediction. The framework distinguishes
geometric LOS labels from standardized TDL/CDL profile proxy labels and marks
synthetic smoke data as debug-only.

## Start here

- [Research quickstart](docs/research/QUICKSTART.md)
- [Research program and evidence limits](docs/research/RESEARCH_PROGRAM.md)
- [Repository architecture](docs/architecture/README.md)
- [Full phased refactor plan](docs/architecture/REFACTOR_PLAN.md)
- [Master 5G NR research and engineering plan](docs/research/MASTER_RESEARCH_PLAN.md)
- [2025 thesis PDF](docs/thesis/NguyenLong.pdf)

## Install and run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m loslab doctor
```

Sionna PHY and ray tracing are optional integrations. Install the Sionna extra
only in an environment with compatible PyTorch, CUDA, Mitsuba, and Dr.Jit; see
the quickstart before running native simulations.

## NR PUSCH channel estimation

The optional Sionna PHY integration includes an initial DMRS-based PUSCH LS
receiver. Install it with <code>python -m pip install -e '.[sionna]'</code> in a
compatible environment. Select a PyTorch build that matches the host CUDA setup
before installing the extra when GPU execution is required.

The receiver accepts a complex resource grid with axes
<code>[batch, rx, rx_antenna, ofdm_symbol, fft_bin]</code> and returns channel
estimates with axes
<code>[batch, rx, rx_antenna, tx, stream, ofdm_symbol, subcarrier]</code>,
Sionna's broadcastable estimator error variance, the DMRS pilot mask, and
configuration/version metadata. Run the native test with
<code>python -m pytest tests/test_integrations/test_nr_pusch_native.py</code>. The native
test is discovered by the regular suite and skips when Sionna is absent. In a
CUDA environment it verifies a known flat complex channel, checks that estimator
NMSE decreases as AWGN variance falls, and exercises a Rayleigh waveform/channel/
receiver loop against Sionna's channel truth. This
golden-vector test validates the adapter contract; synchronization,
channel/noise calibration, BER/BLER, and broad configuration coverage remain
planned work. A separate RT-to-OFDM-grid bridge preserves path delays and maps
Sionna RT CFR onto NR baseband bins; a native 10 m free-space CPU/LLVM test
checks its per-subcarrier phase against the absolute-delay CIR formula. The
complete RT-to-PUSCH receive chain is still under validation.

## Leak-resistant NR records

The versioned NR dataset contracts live in datasets/nr.py. NRObservation
validates grid/estimate axes, FFT-bin mapping, and a typed PUSCH configuration;
its model_inputs method omits sample IDs, targets, and provenance. The Sionna
receiver's <code>observe()</code> method packages estimates with the resource
grid's authoritative effective FFT-bin indices. NRChannelRecord keeps target
values, masks, and provenance outside the model-input payload.

<code>datasets.nr_store</code> writes pickle-free, content-addressed NPZ records
and immutable JSON manifests with SHA256 verification. Use
<code>write_nr_record(root, record)</code> and <code>read_nr_record(root,
manifest)</code> to persist and verify records. Scene-grouped assignments can be
created with <code>evaluation.make_independent_split</code>, audited with
<code>evaluation.audit_group_leakage</code>, and frozen as immutable lock files
with <code>evaluation.freeze_split_lock</code>. These establish storage and split
infrastructure; target semantics, large-scale data generation, and spatial
buffer audits still require research-specific schemas and datasets.

For analytical sanity checks,
<code>datasets.generate_analytical_multipath</code> creates seeded one-to-three
path CFR fixtures with explicit Hz/second units, timing offsets, and a separate
synthetic path oracle. Its direct-component flag is not a geometric LOS label,
and the fixture is not a 3GPP or measured channel model.

<code>propagation.path_steering_diagnostics</code> measures rank, condition
number, and mutual coherence of candidate path steering vectors on a specified
finite set of subcarrier frequencies. It demonstrates exact coincidence and
bandwidth-limited conditioning cases; it is a fixed-delay gain-conditioning
diagnostic, not a nonlinear path estimator, CRLB, or guaranteed resolution
bound.

<code>integrations.sionna.generate_tdl_frequency_response</code> also provides
a pinned Sionna 2.2 TR 38.901 TDL-to-CFR adapter with explicit profile/spec
metadata, centered NR baseband bins, deterministic per-call seeds, and the
original shared RNG state restored after generation. TDL profile names are
recorded as stochastic <em>profile proxies</em>; they are never emitted as
geometric LOS labels. It generates channel coefficients and CFR only, not a
complete PUSCH/CP/synchronization/receiver chain.

For an ideal frequency-domain simulation batch,
<code>integrations.sionna.generate_tdl_pusch_batch</code> connects the PUSCH
transmitter, pinned TDL profile, Sionna OFDM channel, reproducible AWGN, and
PUSCH observation adapter. It keeps transmitted bits and perfect channel truth
separate from <code>NRObservation.model_inputs()</code>. The current link supports
one transmit layer/port, assumes perfect synchronization and a sufficient CP,
and does not model CFO, ICI/ISI, or decode BLER; report it as a simulation
fixture until those receiver checks are implemented.

## Repository map

```text
artifacts/       run identity, manifests, and provenance
configuration/  config loading and canonical hashes
config/          versioned experiment and runtime declarations
datasets/        channel dataset schema and data generators
propagation/     propagation physics and deterministic scene geometry
receivers/       channel observation and pilot estimation
models/          classifier implementations
learning/        training objectives and adaptation algorithms
evaluation/      metrics, group-aware splits, and statistics
experiments/     experiment orchestration
integrations/    optional Sionna adapters
data/ scenes/ cache/ outputs/
notebooks/       active and historical notebooks
archive/         preserved superseded 2025 implementation
tests/           tests organized around active packages
```

The Sionna source checkout used as an API reference is kept separately at
`/home/nguyenl37/sionna`; it is not vendored into this project or imported by
the runtime.

Author: Long Nguyen

# G5 SRS positioning measurements

`integrations.sionna.srs_measurements.measure_ul_srs_rsrp` implements the
bounded UL SRS-RSRP averaging/report contract from ETSI TS 38.215 V19.2.0,
clause 5.2.5. It averages calibrated per-RE power in watts over the caller's
configured SRS resource elements and measurement occasions, retains each
receiver-branch result, and applies a declared selection combiner whose report
is at least the largest individual branch value. The receiver category selects
the corresponding measurement reference point. A versioned JSON Schema is
checked before the report is returned.

Callers supply the per-RE power values, exact SRS mask, occasion and branch
identities, measurement bandwidth, and a SHA-256 digest for the power
calibration artifact. The function binds these declarations but does not
verify the calibration, mapping, bandwidth-to-RE consistency, antenna pattern,
or receiver behavior. Zero power is retained with status
`ZERO_POWER_DIAGNOSTIC` and `null` dBm values.

For mapped multiport SRS grids, `measure_ul_srs_rsrp_from_grid` accepts receive
grids with axes `[occasion, RX branch, slot symbol, subcarrier]`, selects only
REs occupied by the mapped SRS resource, applies a caller-calibrated watts per
grid-unit-squared scale, and binds digests for the received and mapped grids.
It keeps noise and interference power present on those REs, labels the wrapper
`OBSERVED_NOT_QUALIFIED`, and enforces a combined bound on receive-grid and
mapped-resource cells before materialization. It snapshots those inputs so
the recorded digests identify the arrays used for the measurement. This connects the averaging report to an actual mapped
receive-grid representation; calibration, synchronization, interference
separation, and receiver performance still require evidence.

For network-grid observations, `measure_ul_srs_rsrp_from_noisy_network_grid`
accepts a `NetworkGridComposition` together with the `NoisyNetworkGrid` made
from it. It verifies that the selected receiver grid is the coherent
transmitter-contribution sum plus the recorded independent-noise grid, then
binds the selected receiver's noise and per-transmitter contribution digests.
It uses a caller-bounded cell budget and keeps transmitter contributions as
provenance; the reported RSRP is measured on the noisy coherent sum, so
interference cross terms are preserved. This is a synthetic aligned-grid
observation path, not an RF calibration or qualification result.

The same module also reports the TS 38.215 V19.2.0 UL-RTOA reference instant
from SFN/subframe identity and the signed gNB Rx-Tx time difference from
caller-supplied TRP timing observations. Timing reports distinguish receiver
and transmitter reference points and bind a timing-calibration digest. They do
not perform first-path detection, select/verify the closest downlink subframe,
or calibrate a clock. Thus only the timing arithmetic and report semantics are
implemented; timing acquisition remains open.

UL-AoA, path power or TDCT reports, RRC/LPP ASN.1 encoding, synchronized
receive acquisition, and positioning qualification remain open.
The ETSI definition and receiver-specific reference points are in [TS 38.215
V19.2.0, clause 5.2.5](https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/19.02.00_60/ts_138215v190200p.pdf).

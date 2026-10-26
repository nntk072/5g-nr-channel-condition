# G5 NR DL-TDOA semantic report v1

`build_nr_dl_tdoa_measurement_report` composes already-estimated received
subframe-start times into a deterministic, versioned semantic report. It
selects the closest supplied reference-TP estimate for each neighbour and
includes the reference TP itself with zero RSTD. The output records the FR1
UE-antenna-connector or FR2 UE-antenna reference point.

Each TP carries a caller-supplied `NRPRSTimingQuality`. The helper can construct
that value from an uncertainty estimate in metres using an explicitly chosen
standard resolution (0.1, 1, 10, or 30 m). It rounds upward to the selected
resolution and rejects a value that cannot fit the 0..31 field. This is a
conservative quantization of the supplied estimate; it does not estimate or
validate measurement uncertainty.

The JSON shape is defined by
`datasets/schema/nr-dl-tdoa-semantic-report.v1.schema.json`. It preserves
seconds and TP identities for reproducible software exchange. It is not a
TS 37.355 LPP ASN.1/PER encoder: cell/PRS resource identifiers, NR timestamp
coding, optional path and quality extensions, and TS 38.133 RSTD quantization
are outside this contract. It also does not derive subframe starts from PRS
samples, determine first arrival, correct synchronization, or claim positioning
accuracy.

The semantics align with the TS 38.215 DL-RSTD definition (target received
subframe start minus the closest reference received subframe start, with
frequency-range-specific UE reference point) and TS 37.355 NR-DL-TDOA reporting
(reference-TP row included with zero RSTD; NR-TimingQuality represented by a
0..31 value and one of four metre resolutions). See [ETSI TS 38.215
V19.2.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/19.02.00_60/ts_138215v190200p.pdf)
and [ETSI TS 37.355
V19.2.0](https://www.etsi.org/deliver/etsi_ts/137300_137399/137355/19.02.00_60/ts_137355v190200p.pdf).

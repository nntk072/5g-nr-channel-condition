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

This closes only the UL SRS-RSRP arithmetic/report slice. It does not implement
UL-RTOA, gNB Rx-Tx time difference, UL-AoA, path power or TDCT reports, RRC/LPP
ASN.1 encoding, synchronized receive acquisition, or positioning qualification.
The ETSI definition and receiver-specific reference points are in [TS 38.215
V19.2.0, clause 5.2.5](https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/19.02.00_60/ts_138215v190200p.pdf).

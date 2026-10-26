# G5.09 bounded SNR-family acceptance contracts v1 and v2

## Purpose and scope

The independent-channel PUSCH SNR curve can apply a contract supplied before receiver execution. Contract v1 binds the exact SNR grid, channel-seed roster, estimator, and blocks per channel. Timestamped contract v2 adds `preregistered_utc`; its value is included in the exact canonical bytes covered by the detached signature. Both versions set a maximum corrected macro-BLER at each SNR and a minimum paired improvement, defined as uncorrected macro-BLER minus corrected macro-BLER.

The decision uses the curve's simultaneous 95% channel-cluster intervals across clean, uncorrected, and corrected conditions and paired improvements. Transport blocks remain nested within a channel. Every SNR point must satisfy both configured thresholds. Insufficient channel clusters produce NOT_EVALUABLE; they do not pass.

An acceptance contract requires at least 20 independent channel clusters as an executable floor. This floor prevents degenerate tiny-cluster acceptance runs; it is not a powered-sample guarantee, does not establish adequate precision, and does not replace task-specific sample-size planning.

Contract v2's runner records its run-start time before receiver execution and requires the declared preregistration time to precede it. The independent auditor recomputes that ordering from the retained contract and signature metadata. This is a consistency check over declared times: the verifier does not establish a trusted clock, timestamp authority, or actual signing time. Contract v1 remains available for historical signed runs.

This is a conditional result for the synthetic two-path fixture. It does not estimate field BLER, qualify other channel families or receiver configurations, or close G5.09 for deployment.

## Detached signature for a frozen acceptance contract

The SNR-curve CLI verifies a detached OpenSSH signature before constructing the receiver. Sign the exact canonical UTF-8 contract bytes (sorted keys, compact separators, no trailing newline) with namespace `umcd-g5-snr-acceptance-v1`, then supply `--acceptance-contract`, `--acceptance-signature`, `--acceptance-allowed-signers`, and `--acceptance-principal` together. Partial signature options fail closed. Signed v1 runs use curve schema v8; signed timestamped v2 runs use schema v10 and retain the declared preregistration time, receiver start time, chronology status, and an explicit statement that timestamp authority is not established. The independent result auditor can re-verify the exact contract bytes, principal, namespace, and recorded trust/signature digests when given the signature inputs. Historical signed v6 remains readable.

This proves that the supplied allowed-signers file authorizes the principal for the exact frozen contract bytes before receiver execution. It does not prove when the contract was signed, that its declared time is accurate, establish independent reviewer roles, or prove custody of the trust file. Those remain external governance requirements; the receiver outcome remains limited to the synthetic fixture.

## Contract

The v1 input schema is `datasets/schema/g5-ordered-impairments-snr-acceptance.v1.schema.json`; the v2 schema is `datasets/schema/g5-ordered-impairments-snr-acceptance.v2.schema.json`. Threshold arrays must have one entry per SNR point. Contract channel seeds, SNR values, estimator, and block count must exactly match the run request. A v2 contract additionally requires a UTC timestamp ending in `Z`.

~~~json
{
  "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
  "snr_db": [0.0, 5.0, 10.0],
  "channel_seeds": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020],
  "channel_estimator": "sionna_default_ls",
  "blocks_per_channel": 8,
  "minimum_independent_channels": 20,
  "maximum_corrected_macro_bler": [0.2, 0.1, 0.05],
  "minimum_uncorrected_minus_corrected_macro_bler": [0.0, 0.0, 0.0]
}
~~~

For a diagnostic unsigned v1-compatible acceptance run, supply `--independent-channel-count 20 --channel-seed-start 1001 --blocks-per-seed 8 --snr-db 0 5 10 --acceptance-contract contract.json`. Signed acceptance runs also supply all three signature options listed above. A timestamped v2 contract must be signed; it emits schema v10 after the runner confirms the declared timestamp precedes its recorded start. Unsigned v1 acceptance uses schema v7 and signed v1 uses v8. The auditor recomputes the curve intervals and decision from the retained channel/block CRC roster.

Threshold values are task-specific preregistration inputs. The example values are illustrative and are not endorsed acceptance thresholds. Select them from the intended receiver operating requirements before running qualification data.

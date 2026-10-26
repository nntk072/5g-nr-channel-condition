# G5 SNR preregistration and reviewer authorization

The bounded independent-channel SNR curve accepts a signed timestamped criteria contract. Curve schema v11 additionally requires a separately signed reviewer-role policy. These signatures authenticate exact bytes and declared roles; they do not prove trusted time, organizational independence, or trust-file custody.

## Prepare and sign the contract

Create the acceptance contract from the task-specific G5 criteria before selecting or inspecting confirmation outcomes. It must use schema `umcd.g5_ordered_impairments_snr_acceptance.v2`, contain `preregistered_utc` in UTC `...Z` form, and freeze the SNR grid, exact channel roster, estimator, blocks per channel, minimum independent channel count, and simultaneous curve bounds. The contract is canonical compact JSON with sorted keys. Do not use a declared timestamp as a substitute for a trusted timestamping service.

Sign the exact canonical file in the acceptance namespace and provide a separately governed allowed-signers file:

```sh
ssh-keygen -Y sign -f acceptance-reviewer.key \
  -n umcd-g5-snr-acceptance-v1 acceptance-contract.json
```

Create a canonical reviewer-policy JSON document using schema `umcd.g5_snr_reviewer_policy.v1`. It must authorize the acceptance principal exactly once with role `independent_reviewer` and purpose `g5_ordered_impairments_snr_preregistration`. The policy custodian principal and its allowed-signers keyring must be distinct from the acceptance signer and acceptance keyring. Sign it in the policy namespace:

```sh
ssh-keygen -Y sign -f reviewer-policy-custodian.key \
  -n umcd-g5-snr-reviewer-policy-v1 reviewer-policy.json
```

Keep both signed source documents, detached `.sig` files, and the two allowed-signers files with the run record. The application records their digests but cannot establish external custody.

## Run and audit

The receiver CLI requires all four reviewer-policy inputs together with the signed acceptance inputs:

```sh
python -m experiments.g5_ordered_impairments_receiver \
  --independent-channel-count CHANNEL_COUNT \
  --snr-db SNR_1 SNR_2 \
  --acceptance-contract acceptance-contract.json \
  --acceptance-signature acceptance-contract.json.sig \
  --acceptance-allowed-signers acceptance-allowed-signers \
  --acceptance-principal REVIEWER_PRINCIPAL \
  --acceptance-reviewer-policy reviewer-policy.json \
  --acceptance-reviewer-policy-signature reviewer-policy.json.sig \
  --acceptance-reviewer-allowed-signers reviewer-policy-allowed-signers \
  --acceptance-reviewer-policy-principal POLICY_CUSTODIAN \
  --output g5-snr-curve.json
```

Use the exact frozen channel roster and configuration in the contract; the placeholders above are not a recommended sample size or acceptance threshold. Recheck both detached signatures and their role separation with the retained-result auditor:

```sh
python -m experiments.audit_g5_ordered_impairments_result g5-snr-curve.json \
  --acceptance-signature acceptance-contract.json.sig \
  --acceptance-allowed-signers acceptance-allowed-signers \
  --reviewer-policy reviewer-policy.json \
  --reviewer-policy-signature reviewer-policy.json.sig \
  --reviewer-policy-allowed-signers reviewer-policy-allowed-signers \
  --reviewer-policy-principal POLICY_CUSTODIAN
```

The bounded synthetic SNR curve remains diagnostic until task-specific scientific criteria pass with retained evidence and independent review. This admission path alone does not qualify the receiver or release production RT generation or advanced training.

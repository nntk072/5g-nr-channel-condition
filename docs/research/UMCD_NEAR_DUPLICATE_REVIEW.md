# UMCD near-duplicate review workflow

The corpus auditor treats CFR and geometry matches as review candidates. It does not infer leakage from similarity. Report v8 assigns each candidate a content-derived `candidate_id` and binds the sorted candidate roster with `candidate_set_sha256`.

## Review contract

Create a `umcd.near_duplicate_dispositions.v2` JSON object containing:

- `inventory_sha256` and `candidate_set_sha256` copied from the generated report.
- `reviewer_id`, which must equal the OpenSSH principal used to sign the file.
- Exactly one disposition for every reported `candidate_id`, with outcome `legitimate_similarity` or `confirmed_leakage`, a nonblank rationale, and one or more evidence file paths and SHA-256 values.

The exact manifest bytes must be compact, key-sorted UTF-8 JSON followed by one newline. The auditor verifies an OpenSSH detached signature using namespace `umcd-near-duplicate-dispositions-v2@github.com` and a separately supplied reviewer `allowed_signers` file. The reviewer principal and keyring must differ from any verified corpus-inventory or split-lock signer/keyring. Evidence paths must be canonical POSIX relative paths beneath the inventory directory; symlinks and hard-link aliases are rejected. Each file is limited to 64 MiB, with at most 16 files per candidate and 256 MiB total. Stable file identity and every digest are checked before the review is accepted. This verifies file identity and byte binding; the reviewer remains responsible for whether evidence supports the disposition.

## Run and verify

First produce the report. A nonzero exit with `review_required` or `incomplete_comparisons` is expected until the candidate set is reviewed; the report is still written.

```bash
python -m scripts.audit_umcd_corpus /path/to/inventory.json --output /path/to/audit.json
```

After reviewing every candidate, write the signed disposition JSON and signature beside the inventory. The trusted reviewer keyring is supplied separately:

```bash
ssh-keygen -Y sign \
  -n umcd-near-duplicate-dispositions-v2@github.com \
  -f /path/to/reviewer_private_key \
  /path/to/near-duplicate-dispositions.json

python -m scripts.audit_umcd_corpus /path/to/inventory.json \
  --output /path/to/audit-reviewed.json \
  --near-duplicate-dispositions near-duplicate-dispositions.json \
  --near-duplicate-dispositions-signature near-duplicate-dispositions.json.sig \
  --near-duplicate-reviewer-id reviewer-principal \
  --trusted-near-duplicate-reviewers /path/to/externally-governed-reviewers.allowed_signers
```

The disposition and signature paths are relative to the inventory directory. The command exits zero for a complete signed review with no confirmed leakage, or when no cross-split candidates exist. It exits one for unresolved candidates, incomplete comparisons, or confirmed leakage, and two for audit or publication errors. A successful review authenticates the reviewer declaration; it does not establish trust-file custody, evidence truth, complete geometric coverage, or release qualification.

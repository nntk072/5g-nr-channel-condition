# Research result promotion

The promotion workflow separates corpus admission from historical diagnostics.

A promotion policy names the exact result files proposed for admission. scripts/build_result_promotion_manifest.py runs the existing provenance auditor over the complete result root, hashes each requested file from a stable regular-file snapshot, and marks a requested file eligible only when the provenance auditor reports zero issues. A malformed, incomplete, unresolved, or hash-mismatched requested result makes the manifest PROMOTION_REJECTED.

Legacy files that are not requested for promotion remain present and are reported under diagnostic_legacy when they have provenance issues. The workflow never fabricates missing command lines, UTC times, source/config hashes, or referenced artifact bytes, and it never rewrites historical results.

The policy schema is datasets/schema/research-result-promotion-policy.v1.schema.json. The output schema is datasets/schema/research-result-promotion-manifest.v1.schema.json. The manifest remains NOT_QUALIFIED because provenance completeness alone does not establish scientific validity, external trust, or release approval.

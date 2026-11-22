# Shared Finding

A reusable fact records `fact`, `source_sha`, `path`, `symbol_or_line_range`, `verified_by`, `verified_at_utc`, `stale_if`, and `evidence_digest`. Facts are navigation aids and become stale when their source revision or invalidation paths change. `evidence_digest` is the lowercase SHA-256 of the evidence used to establish the fact. Source and tests remain authoritative.

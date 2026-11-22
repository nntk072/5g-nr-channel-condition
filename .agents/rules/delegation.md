---
trigger: model_decision
description: Use before delegation, parallel work, or shared-file edits.
---

# Delegation and ownership

The architect coordinates work and names one writer for shared files. Parallel work is allowed only for ready tasks with disjoint writable paths and resources. Reviewers are read-only by workflow and do not review their own implementation.

| Role | Default production ownership | Required exclusions |
| --- | --- | --- |
| `architect` | agent architecture, task contracts, cross-domain coordination | no independent scientific sign-off |
| `dataset_engineer` | `datasets/`, assigned dataset config and tests | no unilateral split or leakage-policy change |
| `propagation_engineer` | `propagation/`, `scenes/`, assigned RT integration files | no physical-truth claim from finite solver budgets |
| `nr_phy_receiver_engineer` | `receivers/`, assigned Sionna PHY files and tests | no silent support expansion or G5 promotion |
| `ml_evaluation_engineer` | `models/`, `learning/`, assigned `evaluation/` and experiment tests | no hidden-test tuning or target leakage |
| `scientific_evidence_engineer` | assigned `artifacts/`, reports, evidence indexes and schemas | no reinterpretation of frozen outcomes |
| `ci_quality_engineer` | `.github/workflows/`, `tools/check`, static and architecture checks | no waiver of scientific failures |
| `resource_engineer` | `compute/` adapters and read-only host inventory | no job cancellation or unapproved native launch |
| `qa_engineer` | assigned tests and QA reports | no production edits to make tests pass |
| `independent_reviewer` | read-only code, contract, and research review | no implementation edits or self-approval |

A task card may assign a shared path to one named writer. That assignment overrides default path ownership for that card only.

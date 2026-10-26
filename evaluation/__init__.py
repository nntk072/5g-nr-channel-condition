"""Evaluation metrics, grouped splits, duplicate audits, and reporting."""

from .duplicate_audit import (
    CFRDuplicateAudit,
    CFRDuplicatePair,
    TemporalOverlap,
    audit_cfr_near_duplicates,
    find_cross_split_temporal_overlaps,
)
from .g3_joint_evaluation import evaluate_g3_joint_evidence, verify_g3_joint_evidence_bundle
from .g3_joint_evidence_artifact import run_g3_frozen_qualification_evaluation
from .g3_joint_evidence_run_manifest import (
    build_g3_joint_evidence_run_manifest,
    g3_evaluator_source_bundle_identity,
    verify_g3_joint_evidence_run_manifest,
)
from .g3_qualification_contract import (
    build_g3_qualification_run_contract,
    build_g3_qualification_run_contract_v3,
    verify_g3_qualification_run_contract,
)
from .g3_sample_size_planning import (
    compose_g3_joint_sample_size_design,
    verify_g3_joint_sample_size_design,
)
from .g3_receiver_impact import evaluate_receiver_impact
from .g3_path_equivalence_assurance import evaluate_path_equivalence_roster
from .spherical_angular_power import (
    global_to_element_local_directions,
    map_element_frame_spherical_power,
    map_spherical_power,
    spherical_cell_solid_angles,
)
from .independent_splits import (
    IndependentSplit,
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)
from .metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
)
from .spatial_audit import SpatialLeakageAudit, SpatialLeakagePair, audit_spatial_buffer
from .statistics import paired_group_regression_comparison, paired_seed_delta
from .uncertainty import evaluate_grouped_regression_uncertainty
from .uncertainty_io import evaluate_prediction_file

__all__ = [
    "CFRDuplicateAudit",
    "CFRDuplicatePair",
    "IndependentSplit",
    "SpatialLeakageAudit",
    "SpatialLeakagePair",
    "TemporalOverlap",
    "audit_cfr_near_duplicates",
    "audit_group_leakage",
    "audit_spatial_buffer",
    "binary_metrics",
    "block_bootstrap_auc",
    "find_cross_split_temporal_overlaps",
    "evaluate_grouped_regression_uncertainty",
    "evaluate_g3_joint_evidence",
    "run_g3_frozen_qualification_evaluation",
    "verify_g3_joint_evidence_bundle",
    "build_g3_joint_evidence_run_manifest",
    "g3_evaluator_source_bundle_identity",
    "verify_g3_joint_evidence_run_manifest",
    "build_g3_qualification_run_contract",
    "build_g3_qualification_run_contract_v3",
    "verify_g3_qualification_run_contract",
    "compose_g3_joint_sample_size_design",
    "verify_g3_joint_sample_size_design",
    "evaluate_path_equivalence_roster",
    "evaluate_receiver_impact",
    "evaluate_prediction_file",
    "freeze_split_lock",
    "global_to_element_local_directions",
    "map_element_frame_spherical_power",
    "group_split",
    "holdout_domain",
    "in_domain_split",
    "map_spherical_power",
    "make_independent_split",
    "paired_seed_delta",
    "paired_group_regression_comparison",
    "spherical_cell_solid_angles",
]

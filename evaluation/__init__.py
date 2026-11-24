"""Evaluation metrics, grouped splits, duplicate audits, and reporting."""

from datasets.independent_splits import (
    IndependentSplit,
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)

from evaluation.acceptance.joint_evaluation import (
    evaluate_joint_evidence,
    verify_joint_evidence_bundle,
)
from evaluation.acceptance.path_equivalence_assurance import evaluate_path_equivalence_roster
from evaluation.acceptance.qualification_contract import (
    build_qualification_run_contract,
    build_qualification_run_contract_v3,
    verify_qualification_run_contract,
)
from evaluation.acceptance.receiver_impact import evaluate_receiver_impact
from evaluation.audits.duplicate_audit import (
    CFRDuplicateAudit,
    CFRDuplicatePair,
    TemporalOverlap,
    audit_cfr_near_duplicates,
    find_cross_split_temporal_overlaps,
)
from evaluation.audits.population_audit import (
    audit_expected_population,
    validate_expected_population_roster,
)
from evaluation.audits.spatial_audit import (
    SpatialLeakageAudit,
    SpatialLeakagePair,
    audit_spatial_buffer,
)
from evaluation.covariance.spherical_angular_power import (
    global_to_element_local_directions,
    map_element_frame_spherical_power,
    map_spherical_power,
    spherical_cell_solid_angles,
)
from evaluation.joint_evidence_artifact import run_frozen_qualification_evaluation
from evaluation.joint_evidence_run_manifest import (
    build_joint_evidence_run_manifest,
    evaluator_source_bundle_identity,
    verify_joint_evidence_run_manifest,
)
from evaluation.sample_size_planning import (
    compose_joint_sample_size_design,
    verify_joint_sample_size_design,
)
from evaluation.scoring.class_support import (
    ClassSupportCell,
    ClassSupportReport,
    audit_class_support,
)
from evaluation.scoring.low_confidence import build_low_confidence_audit
from evaluation.scoring.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
)
from evaluation.scoring.prior_shift import (
    adjust_probabilities_between_priors,
    adjust_probabilities_for_prior_shift,
    validate_prior_shift_spec,
)
from evaluation.scoring.probability_calibration import (
    apply_platt_calibrator,
    fit_group_balanced_platt_calibrator,
    invert_platt_calibrator,
)
from evaluation.scoring.statistics import paired_group_regression_comparison, paired_seed_delta
from evaluation.scoring.uncertainty import evaluate_grouped_regression_uncertainty
from evaluation.scoring.uncertainty_io import evaluate_prediction_file

__all__ = [
    "CFRDuplicateAudit",
    "CFRDuplicatePair",
    "ClassSupportCell",
    "ClassSupportReport",
    "IndependentSplit",
    "SpatialLeakageAudit",
    "SpatialLeakagePair",
    "TemporalOverlap",
    "audit_cfr_near_duplicates",
    "audit_class_support",
    "audit_expected_population",
    "audit_group_leakage",
    "audit_spatial_buffer",
    "adjust_probabilities_between_priors",
    "adjust_probabilities_for_prior_shift",
    "build_low_confidence_audit",
    "apply_platt_calibrator",
    "binary_metrics",
    "block_bootstrap_auc",
    "find_cross_split_temporal_overlaps",
    "evaluate_grouped_regression_uncertainty",
    "evaluate_joint_evidence",
    "run_frozen_qualification_evaluation",
    "verify_joint_evidence_bundle",
    "build_joint_evidence_run_manifest",
    "evaluator_source_bundle_identity",
    "verify_joint_evidence_run_manifest",
    "build_qualification_run_contract",
    "build_qualification_run_contract_v3",
    "verify_qualification_run_contract",
    "validate_prior_shift_spec",
    "validate_expected_population_roster",
    "compose_joint_sample_size_design",
    "verify_joint_sample_size_design",
    "evaluate_path_equivalence_roster",
    "evaluate_receiver_impact",
    "evaluate_prediction_file",
    "fit_group_balanced_platt_calibrator",
    "freeze_split_lock",
    "global_to_element_local_directions",
    "map_element_frame_spherical_power",
    "group_split",
    "holdout_domain",
    "in_domain_split",
    "invert_platt_calibrator",
    "map_spherical_power",
    "make_independent_split",
    "paired_seed_delta",
    "paired_group_regression_comparison",
    "spherical_cell_solid_angles",
]

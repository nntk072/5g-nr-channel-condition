# ruff: noqa: F401
"""Audit entry points for ordered-impairment result files."""

import argparse
import hashlib
import io
import json
import math
import os
import random
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from evaluation.signed_attestation import verify_openssh_signature_for_bytes
from scipy.stats import beta

from experiments.acceptance import (
    MIN_INDEPENDENT_ACCEPTANCE_CHANNELS,
    SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
    SNR_CURVE_BOOTSTRAP_DRAWS,
    SNR_CURVE_BOOTSTRAP_SEED,
    SNR_EXECUTION_ACCEPTANCE_SCHEMA,
    SNR_REVIEWER_POLICY_PURPOSE,
    SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE,
    SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
    _verify_reviewer_policy,
    validate_preregistration_order,
)

from .constants import (
    _BLOCK_BOUND_REPORT_SCHEMAS,
    _CFO_CHANNEL_TAIL_SCHEMAS,
    _CUDA_CURVE_BASE_VERSIONS,
    _CUDA_DEVICE_REPORT_SCHEMAS,
    _EXACT_SEED_ROSTER_CURVE_SCHEMAS,
    _HEX,
    _MAX_RESULT_BYTES,
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES,
    _PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS,
    _RESULT_SCHEMA,
    _SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
    REPOSITORY_ROOT,
)
from .evidence_files import (
    _curve_report_schema_status,
    _estimator_source_checks,
    _preflight_waveform_archive,
    _read_bounded_evidence_file,
)
from .file import (
    audit_result_file,
    main,
)
from .hashing import (
    _canonical_json_bytes,
    _reject_constant,
    _sha256,
    _unique_object,
)
from .independent import (
    _audit_independent_snr_curve,
    _audit_result,
)
from .preregistration import (
    _declared_preregistration_order_matches,
    _execution_context_matches,
    _inference_seed_roster_consistent,
    _is_sha256,
    _verify_preregistration_signature,
)
from .references import (
    _iq_imbalance_reference,
    _is_finite_nonnegative,
    _is_finite_number,
    _path_sum_reference,
    _positive_integer_delay_reference,
    _same_float,
)
from .replay import (
    _replay_experiment,
    _run_receiver_probe,
)
from .waveform import (
    _cluster_bootstrap_interval,
    _recompute_waveform_diagnostics,
)

__all__ = [
    "REPOSITORY_ROOT",
    "_BLOCK_BOUND_REPORT_SCHEMAS",
    "_CFO_CHANNEL_TAIL_SCHEMAS",
    "_CUDA_CURVE_BASE_VERSIONS",
    "_CUDA_DEVICE_REPORT_SCHEMAS",
    "_EXACT_SEED_ROSTER_CURVE_SCHEMAS",
    "_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE",
    "_HEX",
    "_MAX_RESULT_BYTES",
    "_MAX_WAVEFORM_EVIDENCE_BYTES",
    "_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES",
    "_PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS",
    "_RESULT_SCHEMA",
    "_canonical_json_bytes",
    "_cluster_bootstrap_interval",
    "_curve_report_schema_status",
    "_estimator_source_checks",
    "_declared_preregistration_order_matches",
    "_execution_context_matches",
    "_inference_seed_roster_consistent",
    "_iq_imbalance_reference",
    "_is_finite_nonnegative",
    "_is_finite_number",
    "_is_sha256",
    "_path_sum_reference",
    "_positive_integer_delay_reference",
    "_preflight_waveform_archive",
    "_read_bounded_evidence_file",
    "_recompute_waveform_diagnostics",
    "_reject_constant",
    "_replay_experiment",
    "_run_receiver_probe",
    "_same_float",
    "_sha256",
    "_unique_object",
    "_verify_preregistration_signature",
    "_audit_independent_snr_curve",
    "_audit_result",
    "audit_result_file",
    "main",
]

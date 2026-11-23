"""Schema identifiers, byte budgets and frozen lookup tables for the ordered-impairment audit."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


_RESULT_SCHEMA = "umcd.g5_ordered_impairments_result.v2"


_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE = "umcd-g5-snr-acceptance-v1"


_BLOCK_BOUND_REPORT_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(15, 25)
}


_PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(7, 25)
}


_CFO_CHANNEL_TAIL_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(6, 25)
}


_CUDA_DEVICE_REPORT_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(21, 25)
}


_MAX_RESULT_BYTES = 32 * 1024 * 1024


_HEX = set("0123456789abcdef")


_MAX_WAVEFORM_EVIDENCE_BYTES = 512 * 1024 * 1024


_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


_EXACT_SEED_ROSTER_CURVE_SCHEMAS = {
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v13",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v14",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v15",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v16",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
}


_CUDA_CURVE_BASE_VERSIONS = {12: 9, 13: 7, 14: 8, 15: 10, 16: 11, 25: 17}

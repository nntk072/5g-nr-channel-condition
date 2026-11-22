"""Expected source fingerprints for immutable historical G5 result artifacts."""

from __future__ import annotations

from typing import Any

FROZEN_G5_SOURCE_SHA256 = {
    "experiments/doppler_awgn_receiver.py": "9dc1b6ba19231cce399e8d19b69f3f8aec1345b179790016d7d0ffa5eea8b44a",
    "experiments/independent_channel_receiver.py": "feaaa3f94b1dc25f21be63e8ab51242588fe735212d2253a495249553e18e5fd",
    "experiments/time_varying_multipath_receiver.py": "7e3adbc8a1b2083ecab37db29ac593efd1ae34398232f5bda45ce9b7ff1ec0b7",
    "experiments/qualify_doppler_dmrs_density.py": "44cbd1957648400e851186d9d17f1364d839af4cc95ea51dca027c13bdadd9a1",
    "experiments/scan_doppler_dmrs_density.py": "924bf8f867ba947eef42dc5272fbcbaa3d830d9f126dda496db9f86fc2196d56",
    "experiments/scan_doppler_operating_point.py": "18aba4d8a3e62c90b3aaa35d2c16279a368a2634e829259869668bfa34d29adc",
    "integrations/sionna/nr_pusch.py": "073fb321bd69547478c6c6862a0e355ae1f708f8895a5951423a16ee24f64a41",
    "integrations/sionna/ofdm_impairments.py": "d79e5b6cdd2280cad685bebc904bd6d345ce061621785944b02aff9fff62b751",
    "receivers/nr/pusch.py": "f00a94da21f925c7366a36072fb63dd2dd68669bd2414064e797a97d8fd8464d",
}


def assert_frozen_source_manifest(
    report: dict[str, Any], *, expected_paths: tuple[str, ...]
) -> None:
    """Verify an archived report against frozen hashes, not mutable worktree files."""
    source_hashes = report.get("source_sha256")
    assert isinstance(source_hashes, dict)
    assert set(source_hashes) == set(expected_paths)
    assert all(source_hashes[path] == FROZEN_G5_SOURCE_SHA256[path] for path in expected_paths)

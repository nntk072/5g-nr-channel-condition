"""Deterministic replay of a recorded ordered-impairment experiment."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def _run_receiver_probe(**kwargs: Any) -> dict[str, Any]:
    """Load the optional Torch-backed receiver only for requested replay."""
    from experiments.ordered_impairments_receiver import (
        run_independent_ordered_impairment_probe,
    )

    return run_independent_ordered_impairment_probe(**kwargs)


def _replay_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    config = experiment.get("config")
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object for deterministic replay")
    seeds = config.get("channel_seeds")
    blocks_per_channel = config.get("blocks_per_channel")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) > 64
        or isinstance(blocks_per_channel, bool)
        or not isinstance(blocks_per_channel, int)
        or not 1 <= blocks_per_channel <= 8
    ):
        raise ValueError("deterministic replay is bounded to at most 64 channels × 8 blocks")
    estimator_identity = config.get("channel_estimator", {"name": "sionna_default_ls"})
    if not isinstance(estimator_identity, dict) or estimator_identity.get("name") not in {
        "sionna_default_ls",
        "finite_delay_dft",
    }:
        raise ValueError("deterministic replay requires a supported estimator identity")
    estimator_name = estimator_identity["name"]
    evidence = config.get("waveform_evidence")
    if evidence is None:
        return _run_receiver_probe(
            channel_seeds=tuple(seeds),
            blocks_per_channel=blocks_per_channel,
            channel_estimator=estimator_name,
        )
    if not isinstance(evidence, dict) or not isinstance(evidence.get("file_name"), str):
        raise ValueError("waveform evidence manifest is invalid for deterministic replay")
    with tempfile.TemporaryDirectory(prefix="g5-waveform-replay-") as directory:
        evidence_path = Path(directory) / evidence["file_name"]
        return _run_receiver_probe(
            channel_seeds=tuple(seeds),
            blocks_per_channel=blocks_per_channel,
            waveform_evidence_output=evidence_path,
            channel_estimator=estimator_name,
        )

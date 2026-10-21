"""CPU tests for profile-proxy dataset summaries and manifest labels."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
from datasets.channel import ChannelDataset

ROOT = Path(__file__).resolve().parents[2]
RUNNER = runpy.run_path(str(ROOT / "scripts" / "run_profile_proxy_dataset.py"))
summarize_dataset = RUNNER["summarize_dataset"]


def test_profile_proxy_summary_keeps_family_and_delay_spread_separate():
    data = ChannelDataset(
        h=np.asarray([[[1.0 + 0j] * 8], [[2.0 + 0j] * 8]], dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["cdl", "tdl"]),
        source=np.asarray(["cdl", "tdl"]),
        metadata=(
            {"family": "cdl", "profile": "A", "spread_ns": 30.0},
            {"family": "tdl", "profile": "E", "spread_ns": 100.0},
        ),
    )

    summaries = summarize_dataset(data)

    assert summaries == [
        {
            "family": "cdl",
            "profile": "A",
            "delay_spread_ns": 30.0,
            "samples": 1,
            "profile_proxy_label": 0,
            "mean_observed_cfr_power": 1.0,
            "std_observed_cfr_power": 0.0,
        },
        {
            "family": "tdl",
            "profile": "E",
            "delay_spread_ns": 100.0,
            "samples": 1,
            "profile_proxy_label": 1,
            "mean_observed_cfr_power": 4.0,
            "std_observed_cfr_power": 0.0,
        },
    ]

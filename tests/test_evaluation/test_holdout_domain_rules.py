"""Domain holdout and bootstrap guards: each side of a holdout must contain both labels."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from evaluation.scoring.metrics import block_bootstrap_auc, holdout_domain


def _dataset(domains, labels, groups) -> ChannelDataset:
    count = len(labels)
    return ChannelDataset(
        h=np.ones((count, 1, 8), dtype=np.complex64),
        y=np.asarray(labels),
        group=np.asarray(groups),
        domain=np.asarray(domains),
        source=np.asarray(["synthetic"] * count),
        metadata=tuple({"family": "tdl", "profile": "A", "spread_ns": 30.0} for _ in range(count)),
    )


def test_holdout_domain_needs_both_labels_on_each_side() -> None:
    data = _dataset(
        domains=["holdout", "holdout", "train", "train", "train", "train"],
        labels=[0, 0, 0, 1, 0, 1],
        groups=["g1", "g1", "g2", "g2", "g3", "g3"],
    )
    with pytest.raises(ValueError, match="must both contain LOS/NLOS"):
        holdout_domain(data, "holdout")


def test_unknown_holdout_domain_is_rejected() -> None:
    data = _dataset(
        domains=["a", "a", "b", "b"],
        labels=[0, 1, 0, 1],
        groups=["g1", "g1", "g2", "g2"],
    )
    with pytest.raises(ValueError, match="Unknown/only domain"):
        holdout_domain(data, "missing")


def test_block_bootstrap_needs_two_blocks_and_enough_draws() -> None:
    y = np.asarray([0, 1, 0, 1])
    p = np.asarray([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="Need >=2 blocks"):
        block_bootstrap_auc(y, p, np.asarray(["a", "a", "a", "a"]))
    with pytest.raises(ValueError, match="Need >=2 blocks"):
        block_bootstrap_auc(y, p, np.asarray(["a", "b", "a", "b"]), draws=9)

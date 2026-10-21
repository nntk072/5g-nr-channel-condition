"""Group-disjoint splits, AUROC/calibration and link-block bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from datasets.channel import ChannelDataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


def group_split(y: np.ndarray, group: np.ndarray, test_fraction: float = 0.25, seed: int = 42):
    """Strictly disjoint link/location groups with both labels in each split."""
    y = np.asarray(y)
    group = np.asarray(group)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0,1)")
    if len(set(group)) < 4:
        raise ValueError("At least four distinct groups required")
    for offset in range(100):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=seed + offset
        )
        i, j = next(splitter.split(np.zeros(len(y)), y, group))
        if len(set(y[i])) == 2 and len(set(y[j])) == 2:
            assert not set(group[i]).intersection(group[j])
            return i, j
    raise ValueError("Unable to create two-class, group-disjoint split (check group/label balance)")


def holdout_domain(data: ChannelDataset, domain: str, seed: int = 42, val_fraction: float = 0.2):
    test = np.flatnonzero(data.domain == domain)
    train_all = np.flatnonzero(data.domain != domain)
    if not len(test) or not len(train_all):
        raise ValueError(f"Unknown/only domain {domain!r}")
    if len(set(data.y[test])) < 2 or len(set(data.y[train_all])) < 2:
        raise ValueError("Holdout and remaining training domains must both contain LOS/NLOS")
    if set(data.group[test]) & set(data.group[train_all]):
        raise ValueError("Domain partitions share link IDs; data leakage")
    i, j = group_split(data.y[train_all], data.group[train_all], val_fraction, seed)
    return train_all[i], train_all[j], test


def in_domain_split(
    data: ChannelDataset, seed: int = 42, test_fraction: float = 0.25, val_fraction: float = 0.25
):
    train_all, test = group_split(data.y, data.group, test_fraction, seed)
    i, j = group_split(data.y[train_all], data.group[train_all], val_fraction, seed + 123)
    train, val = train_all[i], train_all[j]
    assert not (set(data.group[train]) & set(data.group[val]))
    assert not (set(data.group[train]) & set(data.group[test]))
    return train, val, test


def threshold_from_validation(y: np.ndarray, score: np.ndarray) -> float:
    """Youden J operating point chosen on validation only; never test labels."""
    from sklearn.metrics import roc_curve

    if len(set(y)) < 2:
        raise ValueError("Need both classes in validation")
    fpr, tpr, t = roc_curve(y, score)
    valid = np.isfinite(t)
    if not valid.any():
        return 0.5
    return float(t[valid][np.argmax(tpr[valid] - fpr[valid])])


def binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or not np.all(np.isfinite(p))
        or not np.all((p >= 0) & (p <= 1))
    ):
        raise ValueError("Invalid probability vector")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes required to compute AUROC")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # ECE uses fixed-width bins, reproducible and explicitly defined.
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        take = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if take.any():
            ece += float(take.mean() * abs(y[take].mean() - p[take].mean()))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10_bins": ece,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "n": int(len(y)),
    }


def block_bootstrap_auc(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, seed: int = 42, draws: int = 400
) -> dict:
    """Resample *groups* (locations/trajectories), not individual correlated frames."""
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2 or draws < 10:
        raise ValueError("Need >=2 blocks, >=10 draws")
    rng = np.random.default_rng(seed)
    ix = [np.flatnonzero(groups == g) for g in unique]
    scores = []
    for _ in range(draws):
        indices = np.concatenate([ix[i] for i in rng.integers(0, len(ix), size=len(ix))])
        if len(set(y[indices])) > 1:
            scores.append(float(roc_auc_score(y[indices], p[indices])))
    if len(scores) < max(10, int(0.25 * draws)):
        raise ValueError("Too few bootstrap replicates contain both classes")
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        "auroc_ci95": [float(lo), float(hi)],
        "resamples_valid": len(scores),
        "resamples_requested": draws,
        "resampling_unit": "link/spatial_block",
    }

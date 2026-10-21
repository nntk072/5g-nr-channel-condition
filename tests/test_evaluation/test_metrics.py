import numpy as np
import pytest
from artifacts.provenance import atomic_json, canonical_hash, package_source_sha256, sha256_file
from datasets.channel import ChannelDataset
from datasets.synthetic import debug_multipath
from evaluation.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
    threshold_from_validation,
)
from evaluation.statistics import paired_seed_delta


def test_group_split_no_leakage():
    d = debug_multipath(n_scenes=6, links_per_scene=16)
    tr, te = group_split(d.y, d.group, seed=3)
    assert not set(d.group[tr]) & set(d.group[te])
    assert set(d.y[tr]) == {0, 1} == set(d.y[te])


def test_in_domain_three_way_no_leakage():
    d = debug_multipath(n_scenes=8, links_per_scene=16)
    tr, v, te = in_domain_split(d, seed=9)
    assert not set(d.group[tr]) & set(d.group[v])
    assert not set(d.group[tr]) & set(d.group[te])
    assert not set(d.group[v]) & set(d.group[te])


def test_holdout_domain_no_leakage():
    d = debug_multipath(n_scenes=6, links_per_scene=16)
    tr, v, te = holdout_domain(d, "debug_scene_3", seed=12)
    assert set(d.domain[te]) == {"debug_scene_3"}
    assert "debug_scene_3" not in set(d.domain[tr])
    assert not set(d.group[tr]) & set(d.group[te])


def test_holdout_unknown_domain_error():
    d = debug_multipath()
    with pytest.raises(ValueError):
        holdout_domain(d, "no_such_scene")


def test_identical_group_across_domains_rejected():
    d = debug_multipath()
    group = d.group.copy()
    group[-1] = group[0]
    bad = ChannelDataset(d.h, d.y, group, d.domain, d.source, d.metadata)
    with pytest.raises(ValueError, match="leakage"):
        holdout_domain(bad, "debug_scene_0")


def test_binary_metrics_perfect():
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 1 and m["auroc"] == 1 and m["fp"] == 0


def test_binary_metrics_bad_prob_rejected():
    with pytest.raises(ValueError):
        binary_metrics(np.array([0, 1]), np.array([-0.1, 0.9]))


def test_threshold_validation():
    t = threshold_from_validation(np.array([0, 0, 1, 1]), np.array([0.05, 0.1, 0.7, 0.9]))
    assert 0.1 < t <= 0.7


def test_bootstrap_link_blocks():
    y = np.array([0, 1] * 12)
    p = np.linspace(0, 1, len(y))
    g = np.repeat(np.arange(12), 2)
    ci = block_bootstrap_auc(y, p, g, draws=60)
    assert 0 <= ci["auroc_ci95"][0] <= ci["auroc_ci95"][1] <= 1
    assert ci["resampling_unit"] == "link/spatial_block"


def test_hash_order_independent():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_atomic_json_and_hash(tmp_path):
    p = tmp_path / "manifest.json"
    atomic_json(p, {"seed": 42})
    assert len(sha256_file(p)) == 64 and "seed" in p.read_text()


def test_paired_seed_delta_zero_with_equal_scores():
    rows = []
    for seed in [11, 23, 47, 59, 71, 83]:
        for method in ["mlp", "coral"]:
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "split": {"test_size": 12},
                    "metrics": {"auroc": 0.6},
                }
            )
    result = paired_seed_delta(rows, "mlp", "coral", draws=40)
    assert result["mean_delta"] == 0.0
    assert result["two_sided_sign_flip_p_value"] == 1.0


def test_paired_seed_delta_consistent_wins():
    rows = []
    for seed in [11, 23, 47, 59, 71, 83]:
        for method, score in [("mlp", 0.55), ("coral", 0.75)]:
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "split": {"test_size": 12},
                    "metrics": {"auroc": score},
                }
            )
    result = paired_seed_delta(rows, "mlp", "coral", draws=40)
    assert abs(result["two_sided_sign_flip_p_value"] - 0.03125) < 1e-8


def test_paired_comparison_split_mismatch_rejected():
    rows = [
        {"seed": 11, "method": "mlp", "split": {"size": 1}, "metrics": {"auroc": 0.5}},
        {"seed": 11, "method": "coral", "split": {"size": 2}, "metrics": {"auroc": 0.6}},
    ]
    with pytest.raises(ValueError, match="split"):
        paired_seed_delta(rows, "mlp", "coral")


def test_code_hash_is_stable():
    one = package_source_sha256()
    assert len(one) == 64 and one == package_source_sha256()

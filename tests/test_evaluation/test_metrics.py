import json

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
from evaluation.paired_link_curves import (
    compare_curves,
    compare_rx_correlation_curves,
    summarize_tdl_profile_sweep,
)
from evaluation.statistics import (
    cluster_binary_error_rate_interval,
    paired_binary_error_comparison,
    paired_cluster_binary_error_comparison,
    paired_cluster_binary_error_family_comparison,
    paired_seed_delta,
)


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


def test_binary_metrics_marks_class_dependent_metrics_undefined_for_single_class():
    metrics = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))

    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert metrics["f1"] is None
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == 1.0
    assert metrics["undefined_metric_reasons"] == {
        "f1": "positive_class_absent",
        "auprc": "positive_class_absent",
        "sensitivity": "positive_class_absent",
        "balanced_accuracy": "both_classes_required",
        "auroc": "both_classes_required",
    }


def test_binary_metrics_bad_prob_rejected():
    with pytest.raises(ValueError):
        binary_metrics(np.array([0, 1]), np.array([-0.1, 0.9]))


def test_binary_metrics_rejects_nonbinary_labels_without_integer_truncation():
    with pytest.raises(ValueError, match="binary vector"):
        binary_metrics(np.array([0.5, 1.0]), np.array([0.2, 0.8]))


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


def test_paired_binary_error_comparison_counts_discordance_and_exact_test():
    result = paired_binary_error_comparison(
        [False, False, True, True],
        [True, False, True, False],
        draws=100,
    )

    assert result["both_error"] == 1
    assert result["control_only_error"] == 1
    assert result["treatment_only_error"] == 1
    assert result["both_success"] == 1
    assert result["treatment_minus_control_bler"] == 0.0
    assert result["treatment_minus_control_goodput_bits_per_codeword"] == 0.0
    assert result["mcnemar_exact_two_sided_p_value"] == 1.0


def test_paired_binary_error_comparison_detects_consistent_degradation():
    result = paired_binary_error_comparison(
        [False] * 5,
        [True] * 5,
        draws=100,
    )

    assert result["treatment_minus_control_bler"] == 1.0
    assert result["paired_bootstrap_ci95"] == [1.0, 1.0]
    assert result["mcnemar_exact_two_sided_p_value"] == pytest.approx(0.0625)


def test_paired_binary_error_comparison_rejects_unmatched_masks():
    with pytest.raises(ValueError, match="equal length"):
        paired_binary_error_comparison([False], [False, True])


def test_paired_cluster_error_interval_resamples_independent_parents_not_rows():
    groups = ["a", "b", "c", "d"]
    control = [False, False, False, False]
    treatment = [True, True, False, False]
    result = paired_cluster_binary_error_comparison(control, treatment, groups, seed=17, draws=2000)

    replicated = paired_cluster_binary_error_comparison(
        [value for value in control for _ in range(10)],
        [value for value in treatment for _ in range(10)],
        [group for group in groups for _ in range(10)],
        seed=17,
        draws=2000,
    )

    assert result["independent_groups"] == 4
    assert result["paired_trials"] == 4
    assert result["treatment_minus_control_error_rate"] == pytest.approx(0.5)
    assert result["cluster_bootstrap_ci95"] == replicated["cluster_bootstrap_ci95"]
    assert replicated["independent_groups"] == 4
    assert replicated["paired_trials"] == 40
    assert result["sampling_unit"].startswith("independent group")


def test_cluster_binary_error_interval_resamples_whole_groups_deterministically():
    errors = [False, True, True, True, False, False]
    groups = ["a", "a", "b", "b", "c", "c"]
    result = cluster_binary_error_rate_interval(errors, groups, seed=71, draws=2000)
    replicated = cluster_binary_error_rate_interval(errors, groups, seed=71, draws=2000)

    assert result == replicated
    assert result["error_rate"] == pytest.approx(0.5)
    assert result["independent_groups"] == 3
    assert 0.0 <= result["cluster_bootstrap_ci95"][0]
    assert result["cluster_bootstrap_ci95"][0] < result["cluster_bootstrap_ci95"][1] <= 1.0
    assert result["status"] == "estimated"


def test_cluster_binary_error_interval_reports_one_group_as_insufficient():
    result = cluster_binary_error_rate_interval([False, True], ["one", "one"], draws=50)

    assert result["independent_groups"] == 1
    assert result["cluster_bootstrap_ci95"] is None
    assert result["status"] == "insufficient_independent_groups"


@pytest.mark.parametrize(
    ("errors", "groups", "message"),
    [
        ([False], [], "equally sized"),
        ([False], [""], "non-empty strings"),
        ([0, 1], ["a", "b"], "boolean values"),
        ([False], ["a"], "positive integer"),
    ],
)
def test_cluster_binary_error_interval_rejects_invalid_inputs(errors, groups, message):
    kwargs = {"draws": 0} if message == "positive integer" else {"draws": 10}
    with pytest.raises(ValueError, match=message):
        cluster_binary_error_rate_interval(errors, groups, **kwargs)


@pytest.mark.parametrize(
    ("control", "treatment", "groups", "message"),
    [
        ([False], [False, True], ["a"], "equally sized"),
        ([False], [False], [""], "non-empty strings"),
        ([False, False], [False, False], ["same", "same"], "at least two independent groups"),
        ([0, 1], [False, True], ["a", "b"], "boolean outcomes"),
    ],
)
def test_paired_cluster_error_comparison_rejects_invalid_inputs(
    control, treatment, groups, message
):
    with pytest.raises(ValueError, match=message):
        paired_cluster_binary_error_comparison(control, treatment, groups)


def test_curve_comparator_requires_matching_bits_and_compares_masks(tmp_path):
    base = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "matched_realization_checks": [
                {
                    "seed": 4,
                    "noise_seed": 6,
                    "transmit_bits_sha256": "same",
                    "channel_truth_sha256": "same-channel",
                }
            ]
        },
        "points": [
            {
                "snr_db": 12.0,
                "frames": 2,
                "bits": 200,
                "mean_transmitted_tb_bits_per_codeword": 100.0,
                "frame_error_mask": [False, False],
            }
        ],
    }
    control = {**base, "configuration": {**base["configuration"], "speed_mps": 3.0}}
    treatment = {
        **base,
        "configuration": {**base["configuration"], "speed_mps": 30.0},
        "points": [
            {
                "snr_db": 12.0,
                "frames": 2,
                "bits": 200,
                "mean_transmitted_tb_bits_per_codeword": 100.0,
                "frame_error_mask": [True, False],
            }
        ],
    }
    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    result = compare_curves(control_path, treatment_path)

    assert result["points"][0]["treatment_minus_control_bler"] == 0.5
    assert result["points"][0]["mcnemar_exact_two_sided_p_value_holm"] == 1.0


def test_curve_comparator_rejects_different_transmitted_bits(tmp_path):
    curve = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "speed_mps": 3.0,
            "matched_realization_checks": [
                {
                    "seed": 4,
                    "noise_seed": 6,
                    "transmit_bits_sha256": "one",
                    "channel_truth_sha256": "same-channel",
                }
            ],
        },
        "points": [{"snr_db": 12.0, "frames": 1, "frame_error_mask": [False]}],
    }
    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    atomic_json(control_path, curve)
    treatment = json.loads(control_path.read_text())
    treatment["configuration"]["speed_mps"] = 30.0
    treatment["configuration"]["matched_realization_checks"][0]["transmit_bits_sha256"] = "two"
    atomic_json(treatment_path, treatment)

    with pytest.raises(ValueError, match="transmitted bits differ"):
        compare_curves(control_path, treatment_path)


def test_curve_comparator_pairs_dmrs_change_on_identical_channel_truth(tmp_path):
    checks = [
        {
            "seed": 4,
            "noise_seed": 6,
            "transmit_bits_sha256": "control-bits",
            "channel_truth_sha256": "same-channel",
        }
    ]
    base = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {"speed_mps": 30.0, "matched_realization_checks": checks},
    }
    control = {
        **base,
        "configuration": {**base["configuration"], "dmrs_additional_position": 0},
        "points": [
            {
                "snr_db": 18.0,
                "frames": 2,
                "bits": 200,
                "mean_transmitted_tb_bits_per_codeword": 100.0,
                "frame_error_mask": [False, False],
            }
        ],
    }
    treatment_checks = [{**checks[0], "transmit_bits_sha256": "treatment-bits"}]
    treatment = {
        **base,
        "configuration": {
            **base["configuration"],
            "dmrs_additional_position": 3,
            "matched_realization_checks": treatment_checks,
        },
        "points": [
            {
                "snr_db": 18.0,
                "frames": 2,
                "bits": 120,
                "mean_transmitted_tb_bits_per_codeword": 60.0,
                "frame_error_mask": [False, True],
            }
        ],
    }
    control_path = tmp_path / "control-dmrs.json"
    treatment_path = tmp_path / "treatment-dmrs.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    result = compare_curves(
        control_path,
        treatment_path,
        varying_config_field="dmrs_additional_position",
    )

    assert result["paired_design"]["same_channel_truth_by_seed"]
    assert not result["paired_design"]["same_transmitted_bits_by_seed"]
    assert result["points"][0]["treatment_minus_control_goodput_bits_per_codeword"] == -70.0


def test_curve_comparator_pairs_explicit_cp_overrun_delay_spread(tmp_path):
    checks = [
        {
            "seed": 51000,
            "noise_seed": 51002,
            "noise_real_seed": 51002,
            "noise_imag_seed": 51003,
            "transmit_bits_sha256": "same-bits",
            "channel_truth_sha256": "control-channel",
        }
    ]
    base_config = {
        "speed_mps": 3.0,
        "matched_realization_checks": checks,
        "transmit_waveform_mean_power_by_batch": [1.0],
        "channel_memory_samples": 13,
        "cp_covers_channel_memory": True,
        "allow_cp_overrun": False,
        "l_max": 7,
    }
    common = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "points": [
            {
                "snr_db": 12.0,
                "frames": 2,
                "bits": 200,
                "frame_error_mask": [False, True],
            }
        ],
    }
    control = {
        **common,
        "configuration": {**base_config, "delay_spread_s": 100e-9},
    }
    treatment = {
        **common,
        "configuration": {
            **base_config,
            "delay_spread_s": 300e-9,
            "channel_memory_samples": 14,
            "cp_covers_channel_memory": False,
            "allow_cp_overrun": True,
            "l_max": 8,
            "matched_realization_checks": [
                {
                    **checks[0],
                    "channel_truth_sha256": "treatment-channel",
                }
            ],
        },
    }
    control_path = tmp_path / "cp-contained.json"
    treatment_path = tmp_path / "cp-overrun.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    result = compare_curves(
        control_path,
        treatment_path,
        varying_config_field="delay_spread_s",
    )

    assert result["paired_design"]["same_transmitted_bits_by_seed"]
    assert not result["paired_design"]["same_channel_truth_by_seed"]
    assert result["control"]["cp_covers_channel_memory"] is True
    assert result["treatment"]["cp_covers_channel_memory"] is False


def test_curve_comparator_pairs_different_tdl_profiles(tmp_path):
    checks = [
        {
            "seed": 51000,
            "noise_seed": 51002,
            "noise_real_seed": 51002,
            "noise_imag_seed": 51003,
            "transmit_bits_sha256": "same-bits",
            "channel_truth_sha256": "tdl-b-channel",
        }
    ]
    base_config = {
        "profile": "TDL-B",
        "speed_mps": 3.0,
        "matched_realization_checks": checks,
        "transmit_waveform_mean_power_by_batch": [1.0],
    }
    common = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "points": [
            {
                "snr_db": 12.0,
                "frames": 2,
                "bits": 200,
                "frame_error_mask": [False, True],
            }
        ],
    }
    control = {**common, "configuration": base_config}
    treatment = {
        **common,
        "configuration": {
            **base_config,
            "profile": "TDL-C",
            "matched_realization_checks": [{**checks[0], "channel_truth_sha256": "tdl-c-channel"}],
        },
    }
    control_path = tmp_path / "tdl-b.json"
    treatment_path = tmp_path / "tdl-c.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    result = compare_curves(control_path, treatment_path, varying_config_field="profile")

    assert result["paired_design"]["same_transmitted_bits_by_seed"]
    assert not result["paired_design"]["same_channel_truth_by_seed"]
    assert result["control"]["profile"] == "TDL-B"
    assert result["treatment"]["profile"] == "TDL-C"


def test_curve_comparator_pairs_estimators_by_batch_cluster(tmp_path):
    checks = [
        {
            "seed": seed,
            "noise_seed": seed + 2,
            "noise_real_seed": seed + 2,
            "noise_imag_seed": seed + 3,
            "transmit_bits_sha256": f"bits-{seed}",
            "channel_truth_sha256": f"channel-{seed}",
        }
        for seed in (41, 42, 43, 44)
    ]
    common = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v2",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "speed_mps": 3.0,
            "matched_realization_checks": checks,
            "transmit_waveform_mean_power_by_batch": [1.0] * len(checks),
        },
    }
    groups = [str(seed) for seed in (41, 42, 43, 44)]
    point_data = [
        {
            "snr_db": 8.0,
            "frames": 4,
            "frame_error_mask": [True, True, False, False],
            "frame_cluster_ids": groups,
            "bits": 400,
        },
        {
            "snr_db": 12.0,
            "frames": 4,
            "frame_error_mask": [False, False, False, False],
            "frame_cluster_ids": groups,
            "bits": 400,
        },
    ]
    control = {
        **common,
        "configuration": {
            **common["configuration"],
            "channel_estimator": {
                "kind": "sionna_ls",
                "min_delay_samples": None,
                "max_delay_samples": None,
            },
        },
        "points": point_data,
    }
    treatment = {
        **common,
        "configuration": {
            **common["configuration"],
            "channel_estimator": {
                "kind": "finite_delay_dft",
                "min_delay_samples": 0,
                "max_delay_samples": 8,
            },
        },
        "points": [
            {**point_data[0], "frame_error_mask": [True, False, False, False]},
            point_data[1],
        ],
    }
    control_path = tmp_path / "ls-curve.json"
    treatment_path = tmp_path / "dft-curve.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    result = compare_curves(
        control_path,
        treatment_path,
        varying_config_field="channel_estimator",
    )

    assert result["schema"] == "loslab.paired_tdl_pusch_time_bler.v3"
    assert result["paired_design"]["same_transmitted_bits_by_seed"] is True
    assert result["paired_design"]["same_channel_truth_by_seed"] is True
    assert result["paired_design"]["sampling_unit"] == (
        "independent batch seed; codewords nested within batch"
    )
    assert result["paired_design"]["interval_scope"] == (
        "pointwise plus simultaneous familywise 95% max-deviation cluster bootstrap across SNR"
    )
    assert result["points"][0]["cluster_paired_comparison"]["independent_groups"] == 4
    family = result["points"][0]["familywise_cluster_comparison"]
    assert family["familywise_snr_points"] == 2
    assert family["familywise_confidence_level"] == 0.95
    assert (
        family["familywise_cluster_bootstrap_ci95"][0]
        <= family["treatment_minus_control_error_rate"]
        <= family["familywise_cluster_bootstrap_ci95"][1]
    )
    assert "mcnemar_exact_two_sided_p_value" not in result["points"][0]


def test_paired_cluster_family_bootstrap_uses_shared_roster_and_bounds_memory():
    groups = ["a", "a", "b", "b", "c", "c"]
    estimates = paired_cluster_binary_error_family_comparison(
        [[False, True, False, True, False, True]] * 2,
        [[True, True, False, False, False, False]] * 2,
        [groups, groups],
        seed=9,
        draws=300,
    )

    assert (
        estimates[0]["familywise_cluster_bootstrap_ci95"]
        == estimates[1]["familywise_cluster_bootstrap_ci95"]
    )
    assert estimates[0]["paired_trials"] == 6
    assert estimates[0]["sampling_unit"] == "independent batch seed; codewords nested"
    with pytest.raises(ValueError, match="same independent cluster roster"):
        paired_cluster_binary_error_family_comparison(
            [[False, True], [False, True]],
            [[True, False], [True, False]],
            [["a", "b"], ["a", "c"]],
            draws=10,
        )
    with pytest.raises(ValueError, match="memory limit"):
        paired_cluster_binary_error_family_comparison(
            [[False, True], [True, False]],
            [[True, False], [False, True]],
            [["a", "b"], ["a", "b"]],
            draws=2_500_001,
        )


def test_estimator_curve_comparison_rejects_unmatched_channel_truth(tmp_path):
    checks = [
        {
            "seed": 41,
            "noise_seed": 43,
            "noise_real_seed": 43,
            "noise_imag_seed": 44,
            "transmit_bits_sha256": "same-bits",
            "channel_truth_sha256": "control-channel",
        },
        {
            "seed": 42,
            "noise_seed": 44,
            "noise_real_seed": 44,
            "noise_imag_seed": 45,
            "transmit_bits_sha256": "same-bits-2",
            "channel_truth_sha256": "control-channel-2",
        },
    ]
    common = {
        "schema": "loslab.tdl_pusch_time_bler_curve.v2",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "speed_mps": 3.0,
            "matched_realization_checks": checks,
            "transmit_waveform_mean_power_by_batch": [1.0, 1.0],
        },
        "points": [
            {
                "snr_db": 10.0,
                "frames": 2,
                "frame_error_mask": [False, True],
                "frame_cluster_ids": ["41", "42"],
            }
        ],
    }
    control = {
        **common,
        "configuration": {
            **common["configuration"],
            "channel_estimator": {"kind": "sionna_ls"},
        },
    }
    treatment_checks = [
        {**checks[0]},
        {**checks[1], "channel_truth_sha256": "different-channel"},
    ]
    treatment = {
        **common,
        "configuration": {
            **common["configuration"],
            "channel_estimator": {"kind": "finite_delay_dft"},
            "matched_realization_checks": treatment_checks,
        },
    }
    control_path = tmp_path / "control-estimator.json"
    treatment_path = tmp_path / "treatment-estimator.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    with pytest.raises(ValueError, match="channel truth differs"):
        compare_curves(
            control_path,
            treatment_path,
            varying_config_field="channel_estimator",
        )


def test_tdl_profile_sweep_requires_matched_trials_and_adjusts_family(tmp_path):
    paths = []
    for letter in "ABCDE":
        curve = {
            "schema": "loslab.tdl_pusch_time_bler_curve.v1",
            "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
            "configuration": {
                "profile": f"TDL-{letter}",
                "speed_mps": 3.0,
                "matched_realization_checks": [
                    {
                        "seed": 1,
                        "noise_seed": 3,
                        "noise_real_seed": 3,
                        "noise_imag_seed": 4,
                        "transmit_bits_sha256": "same-bits",
                        "channel_truth_sha256": f"channel-{letter}",
                    }
                ],
                "transmit_waveform_mean_power_by_batch": [1.0],
            },
            "points": [
                {
                    "snr_db": 12.0,
                    "frames": 4,
                    "bler_errors": int(letter in "CE"),
                    "bler": 0.25 if letter in "CE" else 0.0,
                    "bler_wilson_95_ci": [0.0, 0.5],
                    "bit_errors": 4,
                    "bits": 400,
                    "ber": 0.01,
                    "frame_error_mask": [letter in "CE", False, False, False],
                }
            ],
        }
        path = tmp_path / f"tdl-{letter}.json"
        atomic_json(path, curve)
        paths.append(path)

    result = summarize_tdl_profile_sweep(paths)

    assert set(result["comparisons_vs_tdl_a"]) == {"TDL-B", "TDL-C", "TDL-D", "TDL-E"}
    assert result["paired_design"]["same_transmitted_bits_by_batch"]
    assert result["paired_design"]["different_channel_truth_by_profile"]
    assert all(
        "mcnemar_exact_two_sided_p_value_holm_family20" in comparison["points"][0]
        for comparison in result["comparisons_vs_tdl_a"].values()
    )


def test_receive_correlation_comparison_requires_exactly_matched_first_branch(tmp_path):
    shared_config = {
        "profile": "TDL-A",
        "seed": 7,
        "matched_realization_checks": [
            {
                "seed": 7,
                "noise_real_seed": 9,
                "noise_imag_seed": 10,
                "transmit_bits_sha256": "payload",
                "first_rx_channel_truth_sha256": "first-rx",
                "all_rx_channel_truth_sha256": "latent-channel",
            }
        ],
        "clean_receive_power_by_antenna_per_batch": [[1.0, 1.0]],
    }
    control = {
        "schema": "loslab.tdl_pusch_rx_diversity.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": shared_config,
        "points": [
            {
                "snr_db": 8.0,
                "frames": 4,
                "single_rx": {"frame_error_mask": [True, False, True, False]},
                "two_rx": {"frame_error_mask": [True, False, False, False]},
            }
        ],
    }
    treatment_config = {
        **shared_config,
        "rx_correlation_coefficient": 0.9,
        "rx_correlation_matrix_real": [[1.0, 0.9], [0.9, 1.0]],
        "rx_correlation_matrix_imag": [[0.0, 0.0], [0.0, 0.0]],
        "receive_spatial_correlation_assumption": "correlated",
        "matched_realization_checks": [
            {
                **shared_config["matched_realization_checks"][0],
                "latent_uncorrelated_channel_truth_sha256": "latent-channel",
            }
        ],
        "clean_receive_power_by_antenna_per_batch": [[1.0, 1.0]],
    }
    treatment = {
        **control,
        "configuration": treatment_config,
        "points": [
            {
                **control["points"][0],
                "two_rx": {"frame_error_mask": [True, True, True, False]},
            }
        ],
    }
    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    atomic_json(control_path, control)
    atomic_json(treatment_path, treatment)

    result = compare_rx_correlation_curves(control_path, treatment_path)

    assert result["paired_design"]["same_first_receive_antenna_channel"]
    assert result["points"][0]["treatment_minus_control_bler"] == 0.5
    assert "mcnemar_exact_two_sided_p_value_holm" in result["points"][0]

    treatment_config["matched_realization_checks"][0]["first_rx_channel_truth_sha256"] = (
        "different-first-rx"
    )
    atomic_json(treatment_path, treatment)
    with pytest.raises(ValueError, match="first receive-antenna channel truth differs"):
        compare_rx_correlation_curves(control_path, treatment_path)


def test_code_hash_is_stable():
    one = package_source_sha256()
    assert len(one) == 64 and one == package_source_sha256()

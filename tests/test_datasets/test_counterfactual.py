import numpy as np
import pytest
from datasets.counterfactual import remove_direct_path_component
from datasets.path_oracle import (
    PathOracleTable,
    PathSolveStatus,
    reconstruct_path_oracle_cfr,
)
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    bind_split_children,
    make_split_policy_registry,
)


def _registry():
    policy = SplitPolicy(
        seed="counterfactual-test",
        split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
    )
    parent = SplitSubject(
        subject_id="episode-parent",
        parent_type=SplitParentType.NETWORK_EPISODE,
        source_family="rt.city",
        relations=(
            # The intervention family links all potential descendants to this parent.
            LeakageRelation(LeakageNamespace.COUNTERFACTUAL_FAMILY, "paired-intervention"),
        ),
    )
    other_parents = [
        SplitSubject(
            subject_id=f"independent-{index}",
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="rt.city",
        )
        for index in range(2)
    ]
    registry = make_split_policy_registry([parent, *other_parents], policy=policy)
    return bind_split_children(
        registry,
        [
            SplitChildBinding("base-view", parent.subject_id, "counterfactual_base"),
            SplitChildBinding("ablated-view", parent.subject_id, "counterfactual_math_ablation"),
        ],
    )


def _paths():
    return PathOracleTable(
        link_ids=("link-a", "link-b"),
        frame_indices=np.array([0, 0]),
        rx_port_indices=np.array([0, 0]),
        tx_port_indices=np.array([0, 0]),
        path_offsets=np.array([0, 2, 3]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
        delay_s=np.array([1e-8, 2e-8, 3e-8]),
        complex_gain=np.array([1 + 2j, 0.25 - 0.5j, 0.75 + 0.1j]),
        interaction_order=np.array([0, 1, 1]),
    )


def test_path_oracle_cfr_reconstruction_uses_absolute_delays_and_carrier_offsets():
    paths = _paths()
    frequencies = np.array([-1e6, 0.0, 1e6])
    carrier_hz = 3.5e9

    actual = reconstruct_path_oracle_cfr(paths, frequencies, carrier_hz=carrier_hz)
    expected = np.zeros((2, 3), dtype=np.complex128)
    for group in range(paths.group_count):
        path_slice = paths.group_slice(group)
        expected[group] = np.sum(
            paths.complex_gain[path_slice, None]
            * np.exp(
                -2j * np.pi * paths.delay_s[path_slice, None] * (carrier_hz + frequencies[None, :])
            ),
            axis=0,
        )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert actual.dtype == np.complex128
    assert not actual.flags.writeable

    without_direct = reconstruct_path_oracle_cfr(
        paths,
        frequencies,
        carrier_hz=carrier_hz,
        include_path_mask=paths.interaction_order != 0,
    )
    np.testing.assert_allclose(
        without_direct[0],
        expected[0]
        - paths.complex_gain[0]
        * np.exp(-2j * np.pi * paths.delay_s[0] * (carrier_hz + frequencies)),
    )
    np.testing.assert_array_equal(without_direct[1], expected[1])


def test_path_oracle_cfr_keeps_no_path_as_zero_and_rejects_failed_solves():
    table = PathOracleTable(
        link_ids=("link-empty", "link-solved"),
        frame_indices=np.array([0, 0]),
        rx_port_indices=np.array([0, 0]),
        tx_port_indices=np.array([0, 0]),
        path_offsets=np.array([0, 0, 1]),
        solve_status=(PathSolveStatus.NO_PATH, PathSolveStatus.SOLVED),
        delay_s=np.array([3e-8]),
        complex_gain=np.array([0.5 - 0.25j]),
        interaction_order=np.array([1]),
    )
    cfr = reconstruct_path_oracle_cfr(
        table,
        np.array([-1e6, 0.0, 1e6]),
        carrier_hz=3.5e9,
    )
    np.testing.assert_array_equal(cfr[0], np.zeros(3, dtype=np.complex128))

    failed = PathOracleTable(
        link_ids=("link-failed",),
        frame_indices=np.array([0]),
        rx_port_indices=np.array([0]),
        tx_port_indices=np.array([0]),
        path_offsets=np.array([0, 0]),
        solve_status=(PathSolveStatus.SOLVER_FAILURE,),
        delay_s=np.array([], dtype=np.float64),
        complex_gain=np.array([], dtype=np.complex128),
        interaction_order=np.array([], dtype=np.int64),
    )
    with pytest.raises(ValueError, match=r"failed/unconverged path groups=\[0\]"):
        reconstruct_path_oracle_cfr(failed, np.array([0.0]), carrier_hz=3.5e9)


def test_path_oracle_cfr_rejects_nonpositive_absolute_rf_frequencies():
    with pytest.raises(ValueError, match="positive RF frequencies"):
        reconstruct_path_oracle_cfr(
            _paths(),
            np.array([-3.5e9, 0.0, 3.5e9]),
            carrier_hz=3.5e9,
        )


def test_direct_path_ablation_reuses_same_paths_delays_and_split_parent():
    frequencies = np.array([-1e6, 0.0, 1e6])
    result = remove_direct_path_component(
        _paths(),
        frequency_offset_hz=frequencies,
        carrier_hz=3.5e9,
        registry=_registry(),
        parent_subject_id="episode-parent",
        base_child_id="base-view",
        ablated_child_id="ablated-view",
    )

    assert result.split == _registry().split_for_child("base-view")
    assert result.removed_direct_path_count.tolist() == [1, 0]
    assert (
        result.path_table_sha256
        == remove_direct_path_component(
            _paths(),
            frequency_offset_hz=frequencies,
            carrier_hz=3.5e9,
            registry=_registry(),
            parent_subject_id="episode-parent",
            base_child_id="base-view",
            ablated_child_id="ablated-view",
        ).path_table_sha256
    )
    expected = 0.25 - 0.5j
    np.testing.assert_allclose(
        result.without_direct_cfr[0],
        expected * np.exp(-2j * np.pi * 2e-8 * (3.5e9 + frequencies)),
    )
    np.testing.assert_allclose(result.without_direct_cfr[1], result.full_cfr[1])
    assert not result.full_cfr.flags.writeable
    assert not result.frequency_offset_hz.flags.writeable


def test_direct_path_ablation_rejects_children_from_another_parent():
    registry = _registry()
    with pytest.raises(ValueError, match="declared split parent"):
        remove_direct_path_component(
            _paths(),
            frequency_offset_hz=np.array([0.0]),
            carrier_hz=3.5e9,
            registry=registry,
            parent_subject_id="other-parent",
            base_child_id="base-view",
            ablated_child_id="ablated-view",
        )


def test_direct_path_ablation_rejects_noncanonical_frequency_roster():
    with pytest.raises(ValueError, match="strictly increasing"):
        remove_direct_path_component(
            _paths(),
            frequency_offset_hz=np.array([1.0, 1.0]),
            carrier_hz=3.5e9,
            registry=_registry(),
            parent_subject_id="episode-parent",
            base_child_id="base-view",
            ablated_child_id="ablated-view",
        )


@pytest.mark.parametrize(
    "status",
    (PathSolveStatus.SOLVER_FAILURE, PathSolveStatus.UNCONVERGED),
)
def test_direct_path_ablation_rejects_unsuccessful_source_groups(status):
    paths = _paths()
    failed = PathOracleTable(
        link_ids=paths.link_ids,
        frame_indices=paths.frame_indices,
        rx_port_indices=paths.rx_port_indices,
        tx_port_indices=paths.tx_port_indices,
        path_offsets=np.array([0, 0, 3]),
        solve_status=(status, PathSolveStatus.SOLVED),
        delay_s=paths.delay_s,
        complex_gain=paths.complex_gain,
        interaction_order=paths.interaction_order,
    )

    with pytest.raises(ValueError, match="failed/unconverged path groups=\\[0\\]"):
        remove_direct_path_component(
            failed,
            frequency_offset_hz=np.array([0.0]),
            carrier_hz=3.5e9,
            registry=_registry(),
            parent_subject_id="episode-parent",
            base_child_id="base-view",
            ablated_child_id="ablated-view",
        )


def test_direct_path_ablation_preserves_explicit_no_path_as_finite_budget_zero():
    paths = PathOracleTable(
        link_ids=("link-a", "link-b"),
        frame_indices=np.array([0, 0]),
        rx_port_indices=np.array([0, 0]),
        tx_port_indices=np.array([0, 0]),
        path_offsets=np.array([0, 0, 1]),
        solve_status=(PathSolveStatus.NO_PATH, PathSolveStatus.SOLVED),
        delay_s=np.array([1e-8]),
        complex_gain=np.array([0.5 + 0.25j]),
        interaction_order=np.array([1]),
    )

    result = remove_direct_path_component(
        paths,
        frequency_offset_hz=np.array([-1e6, 0.0, 1e6]),
        carrier_hz=3.5e9,
        registry=_registry(),
        parent_subject_id="episode-parent",
        base_child_id="base-view",
        ablated_child_id="ablated-view",
    )

    assert np.array_equal(result.full_cfr[0], np.zeros(3, dtype=np.complex128))
    assert result.removed_direct_path_count.tolist() == [0, 0]


def test_direct_path_ablation_matches_close_delay_coherent_reference():
    frequencies = np.array([-15e6, 0.0, 15e6])
    delays = np.array([25e-9, 25.001e-9, 80e-9])
    gains = np.array([1.0 + 0.0j, -1.0 + 0.0j, 0.5 + 0.25j])
    paths = PathOracleTable(
        link_ids=("link-a",),
        frame_indices=np.array([0]),
        rx_port_indices=np.array([0]),
        tx_port_indices=np.array([0]),
        path_offsets=np.array([0, len(delays)]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=delays,
        complex_gain=gains,
        interaction_order=np.array([0, 0, 1]),
    )

    result = remove_direct_path_component(
        paths,
        frequency_offset_hz=frequencies,
        carrier_hz=3.5e9,
        registry=_registry(),
        parent_subject_id="episode-parent",
        base_child_id="base-view",
        ablated_child_id="ablated-view",
    )

    contributions = gains[:, None] * np.exp(
        -2j * np.pi * delays[:, None] * (3.5e9 + frequencies[None, :])
    )
    np.testing.assert_allclose(result.full_cfr[0], contributions.sum(axis=0), atol=1e-14)
    np.testing.assert_allclose(result.without_direct_cfr[0], contributions[2], atol=1e-14)
    np.testing.assert_allclose(
        result.full_cfr[0] - result.without_direct_cfr[0], contributions[:2].sum(axis=0), atol=1e-14
    )
    assert result.removed_direct_path_count.tolist() == [2]

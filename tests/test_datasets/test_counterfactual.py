import numpy as np
import pytest
from datasets.counterfactual import remove_direct_path_component
from datasets.path_oracle import PathOracleTable, PathSolveStatus
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


def test_direct_path_ablation_reuses_same_paths_delays_and_split_parent():
    frequencies = np.array([-1e6, 0.0, 1e6])
    result = remove_direct_path_component(
        _paths(),
        frequency_offset_hz=frequencies,
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
            registry=_registry(),
            parent_subject_id="episode-parent",
            base_child_id="base-view",
            ablated_child_id="ablated-view",
        ).path_table_sha256
    )
    expected = 0.25 - 0.5j
    np.testing.assert_allclose(
        result.without_direct_cfr[0], expected * np.exp(-2j * np.pi * 2e-8 * frequencies)
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
            registry=_registry(),
            parent_subject_id="episode-parent",
            base_child_id="base-view",
            ablated_child_id="ablated-view",
        )

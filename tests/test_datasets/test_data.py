import json

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.synthetic import debug_multipath
from propagation.physics import (
    add_estimation_noise,
    frequency_response,
    observable_features,
    subcarrier_frequencies,
)


def test_debug_has_both_classes():
    d = debug_multipath(n_scenes=3, links_per_scene=12)
    assert set(d.y) == {0, 1}
    assert len(d.y) == 36


def test_debug_los_contains_reflections():
    d = debug_multipath(n_scenes=2, links_per_scene=6)
    assert all(m["reflected_paths"] >= 5 for m in d.metadata)
    assert all(m["direct_path"] == bool(y) for m, y in zip(d.metadata, d.y))


def test_debug_deterministic():
    a = debug_multipath(seed=31)
    b = debug_multipath(seed=31)
    assert np.array_equal(a.h, b.h)
    assert np.array_equal(a.y, b.y)


def test_legacy_channel_model_input_view_exposes_only_read_only_cfr():
    data = debug_multipath(n_scenes=2, links_per_scene=4)
    model_inputs = data.model_inputs()

    np.testing.assert_array_equal(model_inputs.observed_cfr, data.h)
    assert not model_inputs.observed_cfr.flags.writeable
    for private_field in (
        "y",
        "group",
        "domain",
        "source",
        "metadata",
        "generation_report",
        "label_semantics",
    ):
        assert not hasattr(model_inputs, private_field)
    with pytest.raises(ValueError, match="read-only"):
        model_inputs.observed_cfr[0, 0, 0] = 0.0


def test_debug_seeds_change_channels():
    a = debug_multipath(seed=1)
    b = debug_multipath(seed=2)
    assert not np.array_equal(a.h, b.h)


def test_debug_invalid_counts():
    with pytest.raises(ValueError):
        debug_multipath(n_scenes=1)
    with pytest.raises(ValueError):
        debug_multipath(links_per_scene=3)


def test_storage_roundtrip(tmp_path):
    d = debug_multipath(n_scenes=3, links_per_scene=10)
    path = tmp_path / "set.npz"
    d.save(path)
    loaded = ChannelDataset.load(path)
    assert np.allclose(loaded.h, d.h)
    assert np.array_equal(loaded.y, d.y)
    assert loaded.metadata == d.metadata
    assert loaded.generation_report == d.generation_report == {}
    assert loaded.label_semantics == "synthetic_debug_direct_component"


def test_split_child_binding_records_child_kind_on_every_parent_row() -> None:
    source = debug_multipath(n_scenes=2, links_per_scene=4)
    parent_id = "episode-parent-1"
    dataset = ChannelDataset(
        source.h[:2],
        source.y[:2],
        source.group[:2],
        source.domain[:2],
        source.source[:2],
        (
            {"split_subject_id": parent_id, "row": 0},
            {"split_subject_id": parent_id, "row": 1},
        ),
    )

    child = dataset.with_split_child_binding("observation-1", "receiver_observation")

    assert [row["split_child_id"] for row in child.metadata] == [
        "observation-1",
        "observation-1",
    ]
    assert all(row["split_child_kind"] == "receiver_observation" for row in child.metadata)
    assert child.generation_report["parent_subject_id"] == parent_id
    assert child.generation_report["row_count"] == 2
    assert dataset.metadata[0] == {"split_subject_id": parent_id, "row": 0}


def test_split_child_binding_rejects_multiple_parents_and_conflicting_rebinding() -> None:
    source = debug_multipath(n_scenes=2, links_per_scene=4)
    dataset = ChannelDataset(
        source.h[:2],
        source.y[:2],
        source.group[:2],
        source.domain[:2],
        source.source[:2],
        (
            {"split_subject_id": "parent-1"},
            {"split_subject_id": "parent-2"},
        ),
    )
    with pytest.raises(ValueError, match="exactly one parent"):
        dataset.with_split_child_binding("observation-1", "receiver_observation")

    single_parent = ChannelDataset(
        source.h[:1],
        source.y[:1],
        source.group[:1],
        source.domain[:1],
        source.source[:1],
        ({"split_subject_id": "parent-1", "split_child_id": "old", "split_child_kind": "kind"},),
    )
    with pytest.raises(ValueError, match="different split child"):
        single_parent.with_split_child_binding("new", "kind")


@pytest.mark.parametrize(
    ("child_id", "child_kind", "parent_id"),
    [
        (" child", "kind", "parent"),
        ("child", "kind ", "parent"),
        ("child", "kind", " parent"),
    ],
)
def test_split_child_binding_rejects_padded_identities(child_id, child_kind, parent_id) -> None:
    source = debug_multipath(n_scenes=2, links_per_scene=4)
    dataset = ChannelDataset(
        source.h[:1],
        source.y[:1],
        source.group[:1],
        source.domain[:1],
        source.source[:1],
        ({"split_subject_id": parent_id},),
    )
    with pytest.raises(ValueError, match="canonical"):
        dataset.with_split_child_binding(child_id, child_kind)


def test_channel_dataset_rejects_padded_split_group_identity() -> None:
    source = debug_multipath(n_scenes=2, links_per_scene=4)
    groups = source.group.copy()
    groups[0] = f" {groups[0]}"
    with pytest.raises(ValueError, match="group IDs must be nonempty canonical strings"):
        ChannelDataset(
            source.h,
            source.y,
            groups,
            source.domain,
            source.source,
            source.metadata,
        )


def test_generation_report_roundtrips_and_derivations_keep_lineage(tmp_path):
    source = debug_multipath(n_scenes=2, links_per_scene=6)
    report = {
        "schema": "loslab.test_generation_report.v1",
        "attempt_counts": {"empty_paths": 3},
    }
    data = ChannelDataset(
        source.h,
        source.y,
        source.group,
        source.domain,
        source.source,
        source.metadata,
        report,
    )
    path = tmp_path / "reported.npz"
    data.save(path)
    loaded = ChannelDataset.load(path)

    assert loaded.generation_report == report
    selected = loaded.select(np.asarray([0, 1]))
    assert selected.generation_report["operation"] == "select"
    assert selected.generation_report["selected_indices"] == [0, 1]
    assert selected.generation_report["parent_generation_report"] == report
    combined = ChannelDataset.concatenate([loaded, loaded])
    assert combined.generation_report["operation"] == "concatenate"
    assert combined.generation_report["components"] == [report, report]


def test_load_legacy_npz_without_generation_report(tmp_path):
    data = debug_multipath(n_scenes=2, links_per_scene=6)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata_json=np.asarray(json.dumps(data.metadata, sort_keys=True)),
    )

    loaded = ChannelDataset.load(path)
    assert loaded.generation_report == {}
    assert loaded.label_semantics == "unspecified_legacy"


def test_dataset_rejects_nans():
    d = debug_multipath(n_scenes=3, links_per_scene=10)
    h = d.h.copy()
    h[0, 0, 0] = np.nan + 1j * np.nan
    with pytest.raises(ValueError, match="finite"):
        ChannelDataset(h, d.y, d.group, d.domain, d.source, d.metadata)


def test_concat_fails_on_mismatch():
    a = debug_multipath(n_subcarriers=64)
    b = debug_multipath(n_subcarriers=32)
    with pytest.raises(ValueError):
        ChannelDataset.concatenate([a, b])


def test_concat_rejects_mixed_binary_label_semantics():
    from dataclasses import replace

    geometric = debug_multipath(n_scenes=2, links_per_scene=4)
    profile_proxy = replace(geometric, label_semantics="profile_proxy")

    with pytest.raises(ValueError, match="Incompatible binary label semantics"):
        ChannelDataset.concatenate([geometric, profile_proxy])


def test_dataset_rejects_label_semantics_that_conflict_with_report():
    from dataclasses import replace

    data = debug_multipath(n_scenes=2, links_per_scene=4)
    with pytest.raises(ValueError, match="disagrees with generation_report"):
        replace(data, generation_report={"label_semantics": "profile_proxy"})


def test_select_preserves_metadata():
    d = debug_multipath()
    x = d.select(np.array([7, 2, 1]))
    assert x.metadata[0] == d.metadata[7]
    assert len(x.y) == 3


def test_no_model_metadata_in_features():
    d = debug_multipath()
    f1 = observable_features(d.h)
    # Deliberately change every label, group and metadata field; features unchanged.
    changed = ChannelDataset(
        d.h,
        1 - d.y,
        np.array(["x" + str(i) for i in range(len(d.y))]),
        np.repeat("arbitrary", len(d.y)),
        d.source,
        tuple({"oracle": int(i)} for i in range(len(d.y))),
    )
    assert np.array_equal(f1, observable_features(changed.h))


def test_feature_gain_invariance():
    d = debug_multipath()
    assert np.allclose(observable_features(d.h), observable_features(d.h * 7j), atol=1e-6)


def test_feature_phase_opt_in():
    d = debug_multipath()
    assert observable_features(d.h, include_phase=True).shape[1] > observable_features(d.h).shape[1]


def test_channel_sum_known():
    f = subcarrier_frequencies(16, 30e3)
    a = np.asarray([1 + 0j, 0.3 + 0.5j])
    tau = np.asarray([0.0, 1e-6])
    result = frequency_response(a, tau, f)
    assert np.allclose(result[8], sum(a))


def test_awgn_finite():
    h = np.ones((8, 1, 32), dtype=np.complex64)
    out = add_estimation_noise(h, 10, np.random.default_rng(1))
    assert out.shape == h.shape and np.all(np.isfinite(out))
    assert not np.array_equal(out, h)

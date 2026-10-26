from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import loslab.scenario_gbsm as command
import pytest


def _config() -> dict[str, Any]:
    return {
        "schema": "umcd.scenario_gbsm_generation.v1",
        "run_id": "gbsm-pilot-001",
        "sample_ids": ["site-a:drop-001", "site-a:drop-002"],
        "scenario": "UMi",
        "seed": 19,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 64,
        "topology": {
            "ut_loc": [[5.0, 4.0, 1.5], [12.0, 8.0, 1.5]],
            "bs_loc": [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]],
            "ut_velocities": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "in_state": [False, True],
            "los": [True, False],
        },
        "generator": {"sampling_frequency_hz": 30_000.0, "scenario_options": {"o2i_model": "low"}},
    }


def test_scenario_gbsm_cli_validates_and_shapes_explicit_drop_topology() -> None:
    native, generator = command._validate_config(_config())

    assert native["ut_loc"] == [[[5.0, 4.0, 1.5]], [[12.0, 8.0, 1.5]]]
    assert native["in_state"] == [[False], [True]]
    assert native["los"] == [[[True]], [[False]]]
    assert generator["scenario_options"]["o2i_model"] == "low"


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value["sample_ids"].append(value["sample_ids"][0]), "unique nonempty IDs"),
        (lambda value: value["topology"]["los"].pop(), "one Boolean per sample"),
        (lambda value: value["generator"].update(unknown_option=True), "unknown fields"),
        (lambda value: value["generator"].update(num_time_steps=10_000), "num_time_steps"),
    ),
)
def test_scenario_gbsm_cli_rejects_malformed_or_unbounded_config(mutate, message: str) -> None:
    config = _config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        command._validate_config(config)


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_scenario_gbsm_config_reader_rejects_file_aliases(
    tmp_path: Path, alias_kind: str
) -> None:
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(_config()), encoding="utf-8")
    alias = tmp_path / "scenario-alias.json"
    if alias_kind == "symlink":
        alias.symlink_to(source)
    else:
        os.link(source, alias)

    with pytest.raises(ValueError, match="single-link regular file"):
        command._read_config(alias)


def test_scenario_gbsm_cli_generates_and_publishes_distinct_run_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "scenario.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    observed: dict[str, Any] = {}
    record = SimpleNamespace(
        sample_id="site-a:drop-001",
        scenario_los=True,
        metadata={"configuration_sha256": "a" * 64},
    )
    second_record = SimpleNamespace(
        sample_id="site-a:drop-002",
        scenario_los=False,
        metadata={"configuration_sha256": "a" * 64},
    )
    run = SimpleNamespace(run_id="gbsm-pilot-001", records=(record, second_record))
    pointer = SimpleNamespace(
        run_id=run.run_id,
        content_sha256="b" * 64,
        relative_path="scenario_gbsm/manifests/example.json",
        sample_ids=("site-a:drop-001", "site-a:drop-002"),
        schema_version=1,
    )
    sample_ids = ("site-a:drop-001", "site-a:drop-002")
    episode_pointers = [
        SimpleNamespace(
            episode_id=sample_id,
            bundle_sha256=digest,
            relative_path=f"episodes/episodes/{sample_id}/bundle.json",
            schema_version=1,
        )
        for sample_id, digest in zip(sample_ids, ("c" * 64, "d" * 64), strict=True)
    ]
    writes: list[tuple[Path, object]] = []

    def fake_generate(**kwargs: Any) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(command, "generate_scenario_gbsm_frequency_response", fake_generate)
    monkeypatch.setattr(command, "scenario_gbsm_run_from_channel", lambda *args, **kwargs: run)
    monkeypatch.setattr(command, "write_scenario_gbsm_run", lambda *args: pointer)
    monkeypatch.setattr(
        command,
        "scenario_gbsm_run_to_episode_bundles",
        lambda _: tuple(
            SimpleNamespace(episode=SimpleNamespace(episode_id=sample_id))
            for sample_id in sample_ids
        ),
    )

    def fake_write_episode_bundle(root: Path, bundle: object) -> SimpleNamespace:
        writes.append((root, bundle))
        return episode_pointers[len(writes) - 1]

    monkeypatch.setattr(command, "write_episode_bundle", fake_write_episode_bundle)

    report = command.generate_and_store(config_path, tmp_path / "store")

    assert observed["topology"]["ut_loc"] == [
        [[5.0, 4.0, 1.5]],
        [[12.0, 8.0, 1.5]],
    ]
    assert report["scenario_los_counts"] == {"los": 1, "nlos": 1}
    assert report["geometric_los_labels_created"] is False
    assert report["status"] == "SYNTHETIC_STANDARDIZED_MODEL_DATA_NOT_QUALIFIED"
    assert report["store_manifest"]["content_sha256"] == "b" * 64
    assert report["schema"] == "umcd.scenario_gbsm_generation_report.v2"
    assert [item["episode_id"] for item in report["episode_store_manifests"]] == [
        "site-a:drop-001",
        "site-a:drop-002",
    ]
    assert len(writes) == 2
    assert all(root == tmp_path / "store" / "episodes" for root, _ in writes)
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/scenario-gbsm-generation-report.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(report)


def test_scenario_gbsm_cli_rejects_episode_bundle_roster_drift_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "scenario.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    records = tuple(
        SimpleNamespace(
            sample_id=sample_id,
            scenario_los=los,
            metadata={"configuration_sha256": "a" * 64},
        )
        for sample_id, los in (
            ("site-a:drop-001", True),
            ("site-a:drop-002", False),
        )
    )
    run = SimpleNamespace(run_id="gbsm-pilot-001", records=records)
    monkeypatch.setattr(command, "generate_scenario_gbsm_frequency_response", lambda **_: object())
    monkeypatch.setattr(command, "scenario_gbsm_run_from_channel", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        command,
        "scenario_gbsm_run_to_episode_bundles",
        lambda _: (
            SimpleNamespace(episode=SimpleNamespace(episode_id="site-a:drop-001")),
            SimpleNamespace(episode=SimpleNamespace(episode_id="site-a:other-drop")),
        ),
    )

    def unexpected_publication(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("roster mismatch must fail before publishing any manifest")

    monkeypatch.setattr(command, "write_episode_bundle", unexpected_publication)
    monkeypatch.setattr(command, "write_scenario_gbsm_run", unexpected_publication)

    with pytest.raises(ValueError, match="exactly cover the scenario run roster"):
        command.generate_and_store(config_path, tmp_path / "store")

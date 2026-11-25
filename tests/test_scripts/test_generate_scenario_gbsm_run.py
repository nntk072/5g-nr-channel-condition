from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import integrations.sionna.scenario_gbsm_run as command
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
def test_scenario_gbsm_config_reader_rejects_file_aliases(tmp_path: Path, alias_kind: str) -> None:
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "valid UTF-8 JSON"),
        (b"[]", "root must be an object"),
        (b'{"schema":"first","schema":"second"}', "duplicate key 'schema'"),
    ],
)
def test_scenario_gbsm_config_reader_rejects_invalid_json_bytes(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "scenario.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        command._read_config(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("seed"), "missing or unknown root fields"),
        (lambda value: value.update(unexpected=True), "missing or unknown root fields"),
        (lambda value: value.update(schema="future"), "unsupported scenario-GBSM"),
    ],
)
def test_scenario_gbsm_config_reader_rejects_root_contract_drift(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    path = tmp_path / "scenario.json"
    config = _config()
    mutate(config)
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        command._read_config(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(sample_ids=[]), "unique nonempty IDs"),
        (lambda value: value.update(sample_ids="drop-a"), "unique nonempty IDs"),
        (lambda value: value.update(sample_ids=[" drop-a "]), "unique nonempty IDs"),
        (lambda value: value.update(sample_ids=["drop-a", "drop-b"] * 65), "unique nonempty IDs"),
        (lambda value: value.update(run_id=" "), "run_id must be a nonempty string"),
        (lambda value: value.update(run_id=" run "), "run_id must be a nonempty string"),
        (lambda value: value.update(scenario="unknown"), "scenario must be one of"),
        (lambda value: value.update(scenario=1), "scenario must be one of"),
        (lambda value: value.update(seed=True), "seed must be a nonnegative integer"),
        (lambda value: value.update(seed=-1), "seed must be a nonnegative integer"),
        (lambda value: value.update(carrier_frequency_hz=float("inf")), "finite and positive"),
        (lambda value: value.update(subcarrier_spacing_hz=True), "finite and positive"),
        (lambda value: value.update(carrier_frequency_hz=0), "finite and positive"),
        (lambda value: value.update(num_subcarriers=True), "num_subcarriers must be"),
        (lambda value: value.update(num_subcarriers=5000), "num_subcarriers must be"),
    ],
)
def test_scenario_gbsm_validator_rejects_invalid_run_metadata(mutate: Any, message: str) -> None:
    config = _config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        command._validate_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(topology=[]), "topology does not match"),
        (lambda value: value["topology"].pop("ut_loc"), "topology does not match"),
        (lambda value: value["topology"].update(unexpected=[]), "topology does not match"),
        (lambda value: value["topology"].update(ut_loc=[[1.0, 2.0]]), "finite 3-vector"),
        (
            lambda value: value["topology"].update(bs_loc=[[1.0, 2.0, float("nan")], [0, 0, 0]]),
            "finite 3-vector",
        ),
        (
            lambda value: value["topology"].update(ut_velocities=[[True, 0, 0], [0, 0, 0]]),
            "finite 3-vector",
        ),
        (lambda value: value["topology"].update(in_state=[1, True]), "one Boolean per sample"),
        (lambda value: value["topology"].update(los=[True]), "one Boolean per sample"),
    ],
)
def test_scenario_gbsm_validator_rejects_invalid_topology(mutate: Any, message: str) -> None:
    config = _config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        command._validate_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(generator=[]), "generator has unknown fields"),
        (lambda value: value["generator"].update(num_time_steps=True), "num_time_steps must be"),
        (lambda value: value["generator"].update(num_time_steps=0), "num_time_steps must be"),
        (
            lambda value: value["generator"].update(sampling_frequency_hz=None),
            "sampling_frequency_hz",
        ),
        (lambda value: value["generator"].update(sampling_frequency_hz=0), "sampling_frequency_hz"),
        (
            lambda value: value["generator"].update(sampling_frequency_hz=float("nan")),
            "sampling_frequency_hz",
        ),
        (
            lambda value: _expand_to_maximum_grid(value),
            "exceeds 2000000 elements",
        ),
        (lambda value: value["generator"].update(ut_array=[]), "ut_array must be an object"),
        (
            lambda value: value["generator"].update(ut_array={"num_rows": 257}),
            "at most 256 antenna ports",
        ),
        (
            lambda value: value["generator"].update(scenario_options=[]),
            "scenario_options must be an object",
        ),
    ],
)
def test_scenario_gbsm_validator_rejects_invalid_generator_bounds(
    mutate: Any, message: str
) -> None:
    config = _config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        command._validate_config(config)


def test_scenario_gbsm_validator_shapes_optional_orientations() -> None:
    config = _config()
    orientations = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    config["topology"].update(ut_orientations=orientations, bs_orientations=orientations)

    native, _ = command._validate_config(config)

    assert native["ut_orientations"] == [[row] for row in orientations]
    assert native["bs_orientations"] == [[row] for row in orientations]


def _expand_to_maximum_grid(config: dict[str, Any]) -> None:
    sample_ids = [f"drop-{index}" for index in range(128)]
    config.update(sample_ids=sample_ids, num_subcarriers=4096)
    for name, rows in config["topology"].items():
        config["topology"][name] = [rows[0] for _ in sample_ids]
    config["generator"].update(num_time_steps=256)


def test_scenario_gbsm_generator_rejects_episode_manifest_roster_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "scenario.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    records = tuple(
        SimpleNamespace(
            sample_id=sample_id, scenario_los=False, metadata={"configuration_sha256": "a" * 64}
        )
        for sample_id in ("site-a:drop-001", "site-a:drop-002")
    )
    run = SimpleNamespace(run_id="gbsm-pilot-001", records=records)
    monkeypatch.setattr(command, "generate_scenario_gbsm_frequency_response", lambda **_: object())
    monkeypatch.setattr(command, "scenario_gbsm_run_from_channel", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        command,
        "scenario_gbsm_run_to_episode_bundles",
        lambda _: tuple(
            SimpleNamespace(episode=SimpleNamespace(episode_id=sample_id))
            for sample_id in ("site-a:drop-001", "site-a:drop-002")
        ),
    )
    monkeypatch.setattr(
        command, "write_episode_bundle", lambda *args: SimpleNamespace(episode_id="other")
    )

    with pytest.raises(ValueError, match="published episode manifests must exactly cover"):
        command.generate_and_store(config_path, tmp_path / "store")


@pytest.mark.parametrize("raises", (False, True))
def test_scenario_gbsm_cli_main_reports_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], raises: bool
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["generate-scenario-gbsm", "--config", "config.json", "--output-dir", "store"]
    )
    if raises:
        monkeypatch.setattr(
            command,
            "generate_and_store",
            lambda *_: (_ for _ in ()).throw(ValueError("bad config")),
        )
        with pytest.raises(SystemExit) as error:
            command.main()
        assert error.value.code == 2
        assert "error: bad config" in capsys.readouterr().err
    else:
        monkeypatch.setattr(command, "generate_and_store", lambda *_: {"status": "synthetic"})
        command.main()
        assert json.loads(capsys.readouterr().out) == {"status": "synthetic"}

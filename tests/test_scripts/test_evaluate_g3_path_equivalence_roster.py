from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evaluation.g3_joint_acceptance import build_g3_joint_acceptance_family_v4
from scripts.evaluate_g3_path_equivalence_roster import main

from evaluation import g3_path_equivalence_roster_cli as roster_cli_implementation


def _family() -> dict[str, Any]:
    return build_g3_joint_acceptance_family_v4(
        family_id="g3-v4",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("scene-a", "LOS")],
        propagation_expected_strata=[("scene-a", "LOS", "total")],
        receiver_impact_expected_strata=[("scene-a", "LOS")],
        path_equivalence_expected_scenarios=[("scene-a", "LOS", "episode-1")],
    )


def _input() -> dict[str, Any]:
    family = _family()
    matched = [
        {
            "candidate_receiver_relevance": {"receiver_relevant": True},
            "challenger_receiver_relevance": {"receiver_relevant": True},
            "geometry_and_channel_deltas": {
                "maximum_interaction_vertex_displacement_m": 0.0,
                "absolute_delay_difference_s": 0.0,
                "maximum_endpoint_direction_difference_rad": 0.0,
                "aggregate_class_cfr_e_h": 0.0,
            },
        }
        for _ in range(19)
    ]
    comparison = {
        "schema": "loslab.g3_path_equivalence_comparison.v1",
        "status": "PATH_COVERAGE_DIAGNOSTIC_NOT_QUALIFIED",
        "admission_possible": False,
        "class_coverage_status": "PASS",
        "coverage_criterion": 0.95,
        "candidate_relevant_class_count": 20,
        "challenger_relevant_class_count": 19,
        "candidate_bidirectional_coverage": 0.95,
        "challenger_bidirectional_coverage": 1.0,
        "matched_class_pair_count": 19,
        "matched_classes": matched,
        "candidate_only_relevant_classes": [
            {"receiver_relevance": {"receiver_relevant": True}}
        ],
        "challenger_only_relevant_classes": [],
    }
    return {
        "schema": "loslab.g3_path_equivalence_roster_input.v1",
        "joint_acceptance_family": family,
        "records": [
            {
                "family_id": "scene-a",
                "condition": "LOS",
                "episode_id": "episode-1",
                "comparison": comparison,
            }
        ],
    }


def test_cli_emits_roster_report_bound_to_the_validated_family(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "roster.json"
    input_path.write_text(json.dumps(_input()), encoding="utf-8")

    assert main([str(input_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ROSTER_PATH_COVERAGE_DIAGNOSTIC_PASS"
    assert report["admission_possible"] is False
    assert report["joint_acceptance_family"] == {
        "family_id": "g3-v4",
        "family_sha256": _input()["joint_acceptance_family"]["family_sha256"],
    }


def test_cli_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"wrong","schema":"wrong"}', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(path)])


def test_cli_rejects_excessively_nested_json_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "nested.json"
    path.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(path)])


def test_cli_rejects_symlink_input(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_input()), encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(source)

    with pytest.raises(SystemExit, match="2"):
        main([str(link)])


def test_input_schema_accepts_the_frozen_cli_contract() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g3-path-equivalence-roster-input.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_input())


def test_cli_enforces_input_size_cap_before_json_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversized.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(roster_cli_implementation, "_MAX_INPUT_BYTES", 1)

    with pytest.raises(SystemExit, match="2"):
        main([str(path)])


def test_cli_rejects_hardlinked_input(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_input()), encoding="utf-8")
    link = tmp_path / "hardlink.json"
    link.hardlink_to(path)

    with pytest.raises(SystemExit, match="2"):
        main([str(path)])

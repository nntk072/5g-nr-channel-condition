from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest
from datasets.scene_families import (
    FAMILY_IDS,
    load_scene_family_registry,
    sample_scene_candidates,
    scene_candidate_split_metadata,
    scene_candidate_split_relations,
    scene_family_registry_sha256,
    validate_scene_family_registry,
)
from datasets.split_policy import SplitParentType, SplitSubject, derive_split_components
from scripts import sample_scene_families as scene_family_cli
from scripts.sample_scene_families import main as sample_scene_families_main


def _subjects(registry: dict[str, object], candidate_ids: list[str]) -> list[SplitSubject]:
    relations = scene_candidate_split_relations(registry, candidate_ids)
    rows = {row["candidate_id"]: row for row in registry["candidates"]}
    return [
        SplitSubject(
            subject_id=candidate_id,
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family=rows[candidate_id]["family_id"],
            relations=relations[candidate_id],
        )
        for candidate_id in candidate_ids
    ]


def test_scene_registry_schema_hash_and_odbl_release_states() -> None:
    registry = load_scene_family_registry()
    schema = json.loads(
        Path("datasets/schema/scene-family-registry.v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(registry)
    sample_schema = json.loads(
        Path("datasets/schema/scene-family-sample.v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(sample_schema)
    jsonschema.Draft202012Validator(sample_schema).validate(
        sample_scene_candidates(registry, seed=41)
    )

    assert tuple(row["family_id"] for row in registry["taxonomy"]) == FAMILY_IDS
    assert scene_family_registry_sha256(registry) == scene_family_registry_sha256(
        copy.deepcopy(registry)
    )
    odbl = [row for row in registry["candidates"] if row["source_kind"] == "odbl_osm_scene"]
    assert {row["scene_id"] for row in odbl} == {"etoile", "florence", "munich"}
    assert all(row["internal_research_status"] == "allowed" for row in odbl)
    assert all(row["public_release_status"] == "not_cleared" for row in odbl)
    assert all(row["qualification_status"] == "NOT_QUALIFIED" for row in registry["candidates"])


def test_scene_family_sampling_is_deterministic_and_reports_shortfalls() -> None:
    registry = load_scene_family_registry()
    first = sample_scene_candidates(registry, seed=7103, candidates_per_family=2)
    reordered = copy.deepcopy(registry)
    reordered["candidates"].reverse()
    replay = sample_scene_candidates(reordered, seed=7103, candidates_per_family=2)

    assert first == replay
    assert len(first["family_coverage"]) == 12
    by_family = {row["family_id"]: row for row in first["family_coverage"]}
    assert by_family["S00-analytic"]["status"] == "UNSUPPORTED_NO_CANDIDATE"
    assert by_family["S01-urban-canyon"]["available_independence_groups"] == 1
    assert by_family["S01-urban-canyon"]["shortfall"] == 1
    assert by_family["S02-dense-urban"]["available_independence_groups"] == 3
    assert by_family["S03-suburban-rural"]["status"] == "UNSUPPORTED_NO_CANDIDATE"
    assert by_family["S03-suburban-rural"]["shortfall"] == 2
    assert "fixture:simple_street_canyon" not in {
        row["candidate_id"] for row in first["selected_candidates"]
    }
    assert first["qualification_status"] == "NOT_QUALIFIED"


def test_existing_split_relations_group_procedural_variants_and_exact_assets() -> None:
    registry = load_scene_family_registry()
    clone = copy.deepcopy(registry["candidates"][0])
    clone["candidate_id"] = "fixture:simple_street_canyon-copy"
    registry["candidates"].append(clone)
    candidate_ids = [
        "procedural:city41",
        "procedural:city42",
        "procedural:city43",
        "fixture:simple_street_canyon",
        "fixture:simple_street_canyon-copy",
        "odbl:etoile",
        "odbl:florence",
        "odbl:munich",
    ]
    components = derive_split_components(_subjects(registry, candidate_ids))
    component_ids = [{subject.subject_id for subject in component} for component in components]

    assert {"procedural:city41", "procedural:city42", "procedural:city43"} in component_ids
    assert {"fixture:simple_street_canyon", "fixture:simple_street_canyon-copy"} in component_ids
    assert {"odbl:etoile"} in component_ids
    assert {"odbl:florence"} in component_ids
    assert {"odbl:munich"} in component_ids


def test_scene_candidate_metadata_binds_loaded_assets_and_release_policy() -> None:
    registry = load_scene_family_registry()
    city = next(row for row in registry["candidates"] if row["candidate_id"] == "procedural:city41")
    metadata = scene_candidate_split_metadata(
        registry,
        candidate_id="procedural:city41",
        actual_geometry_asset_sha256=city["geometry_asset_sha256"],
    )
    assert metadata["family_id"] == "S01-urban-canyon"
    assert metadata["qualification_status"] == "NOT_QUALIFIED"
    assert metadata["split_leakage_relations"]

    with pytest.raises(ValueError, match="geometry digest differs"):
        scene_candidate_split_metadata(
            registry,
            candidate_id="procedural:city41",
            actual_geometry_asset_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="not admitted"):
        scene_candidate_split_metadata(
            registry,
            candidate_id="fixture:simple_street_canyon",
            actual_geometry_asset_sha256=next(
                row["geometry_asset_sha256"]
                for row in registry["candidates"]
                if row["candidate_id"] == "fixture:simple_street_canyon"
            ),
        )


def test_registry_rejects_invalid_odbl_policy_and_missing_family() -> None:
    registry = load_scene_family_registry()
    invalid_odbl = copy.deepcopy(registry)
    next(row for row in invalid_odbl["candidates"] if row["candidate_id"] == "odbl:etoile")[
        "public_release_status"
    ] = "review_required"
    with pytest.raises(ValueError, match="ODbL scene candidate"):
        validate_scene_family_registry(invalid_odbl)

    missing_family = copy.deepcopy(registry)
    missing_family["taxonomy"].pop()
    with pytest.raises(ValueError, match="complete ordered S00–S11"):
        validate_scene_family_registry(missing_family)


def test_scene_family_cli_retains_shortfalls_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "scene-sample.json"
    assert (
        sample_scene_families_main(
            ["--seed", "410", "--candidates-per-family", "1", "--output", str(output)]
        )
        == 2
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert len(report["family_coverage"]) == 12

    with pytest.raises(SystemExit):
        sample_scene_families_main(
            ["--seed", "410", "--candidates-per-family", "1", "--output", str(output)]
        )


def test_scene_family_cli_reports_registry_validation_errors(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        scene_family_cli,
        "load_scene_family_registry",
        lambda _path: (_ for _ in ()).throw(ValueError("registry fixture rejected")),
    )

    with pytest.raises(SystemExit) as error:
        sample_scene_families_main(
            ["--seed", "410", "--output", str(tmp_path / "report.json")]
        )

    assert error.value.code == 2
    assert "registry fixture rejected" in capsys.readouterr().err


def test_scene_family_cli_handles_no_replace_race(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        scene_family_cli,
        "_publish_no_replace",
        lambda *_args: (_ for _ in ()).throw(FileExistsError("concurrent writer")),
    )

    with pytest.raises(SystemExit) as error:
        sample_scene_families_main(
            ["--seed", "410", "--output", str(tmp_path / "report.json")]
        )

    assert error.value.code == 2
    assert "refusing to overwrite existing report" in capsys.readouterr().err

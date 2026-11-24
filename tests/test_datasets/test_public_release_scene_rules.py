"""Public release manifest: scene assessment, license-link and artifact-roster rules."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from datasets.public_release import validate_public_release_manifest
from tests.test_datasets.test_public_release import _manifest


def _with(mutate) -> dict[str, Any]:
    manifest = copy.deepcopy(_manifest(b"payload"))
    mutate(manifest)
    return manifest


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda m: m.update(artifacts=[]),
            "must enumerate at least one artifact",
            id="no-artifacts",
        ),
        pytest.param(lambda m: m.update(scenes="x"), "scenes must be a list", id="scenes-type"),
        pytest.param(
            lambda m: m["scenes"][0].update(scene_id="other"),
            "does not match the audited OSM asset hash",
            id="scene-hash",
        ),
        pytest.param(
            lambda m: m["scenes"][0].update(license_url="https://example.org/license"),
            "must link directly to ODbL",
            id="license-url",
        ),
        pytest.param(
            lambda m: m["scenes"][0].update(copyright_notice_url="https://example.org/c"),
            "copyright notice",
            id="copyright-url",
        ),
        pytest.param(
            lambda m: m["scenes"][0].update(required_credit="Someone"),
            "required OSM credit",
            id="credit",
        ),
        pytest.param(
            lambda m: m["scenes"][0].update(osm_extract_timestamp="2026-01-01T00:00:00"),
            "must include a timezone",
            id="timezone",
        ),
        pytest.param(
            lambda m: m["scenes"][0].update(conversion_recipe_sha256="ABC"),
            "lowercase SHA-256",
            id="recipe-digest",
        ),
        pytest.param(
            lambda m: m["scenes"][0].update(source_software_revision="b" * 40),
            "software revision differs",
            id="revision",
        ),
    ],
)
def test_scene_assessments_are_bound_to_their_audited_provenance(mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_public_release_manifest(_with(mutate))


def test_scene_assessment_objects_and_roster_are_checked() -> None:
    with pytest.raises(ValueError, match="each public release scene assessment must be an object"):
        validate_public_release_manifest(_with(lambda m: m.update(scenes=["x"])))
    with pytest.raises(ValueError, match="every ODbL scene assessment must be referenced"):
        validate_public_release_manifest(
            _with(
                lambda m: m["artifacts"][0].update(
                    scene_ids=[], scene_classifications={}, source_scene_tree_sha256=[]
                )
            )
        )


def test_artifacts_must_reference_only_declared_scenes() -> None:
    with pytest.raises(ValueError, match="references an undeclared scene"):
        validate_public_release_manifest(
            _with(
                lambda m: m["artifacts"][1].update(
                    scene_ids=["ghost"], scene_classifications={"ghost": "odbl_database"}
                )
            )
        )


def test_artifact_paths_must_be_unique() -> None:
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        validate_public_release_manifest(
            _with(lambda m: m["artifacts"].append(copy.deepcopy(m["artifacts"][0])))
        )


def test_third_party_asset_paths_must_not_repeat() -> None:
    def mutate(manifest):
        assets = manifest["scenes"][0]["third_party_license_closure"]["assets"]
        assets.append(copy.deepcopy(assets[0]))

    with pytest.raises(ValueError, match="repeats a third-party asset path"):
        validate_public_release_manifest(_with(mutate))

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from scripts.build_implementation_feature_map import build_feature_map


def test_feature_map_covers_checklist_and_does_not_promote_qualification() -> None:
    root = Path(__file__).resolve().parents[2]
    backlog = root / "docs/research/IMPLEMENTATION_BACKLOG.md"
    text = backlog.read_text(encoding="utf-8")
    result = build_feature_map(text, root=root)

    assert result["summary"]["feature_count"] == sum(
        line.lstrip().startswith("- [x]") or line.lstrip().startswith("- [ ]")
        for line in text.splitlines()
    )
    assert result["summary"]["feature_count"] == (
        result["summary"]["marked_done_count"] + result["summary"]["open_count"]
    )
    assert result["summary"]["feature_count"] == (
        result["summary"]["linked_count"] + result["summary"]["unmapped_count"]
    )
    assert all(
        row["qualification_status"] == "not_assessed_by_inventory" for row in result["features"]
    )
    allowed_statuses = {"code_complete", "partial", "absent", "externally_blocked"}
    assert {row["implementation_status"] for row in result["features"]} <= allowed_statuses
    assert (
        sum(result["summary"]["implementation_status_counts"].values())
        == result["summary"]["feature_count"]
    )
    assert all(
        row["implementation_status"] == "code_complete"
        for row in result["features"]
        if row["backlog_status"] == "marked_done"
    )
    schema = json.loads(
        (root / "datasets/schema/implementation-feature-map.v3.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)


def test_feature_map_classifies_all_implementation_states_conservatively(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/tool.py").write_text("pass\n")
    text = (
        "### 0. Package\n"
        "- [x] Finished feature.\n"
        "- [ ] Partial implementation in `scripts/tool.py`.\n"
        "- [ ] **Partial:** implementation exists but evidence is not linked here.\n"
        "- [ ] No implementation evidence yet.\n"
        "- [ ] **Externally blocked:** requires independent authority.\n"
    )

    features = build_feature_map(text, root=tmp_path)["features"]

    assert [feature["implementation_status"] for feature in features] == [
        "code_complete",
        "partial",
        "partial",
        "absent",
        "externally_blocked",
    ]
    assert [feature["implementation_status_basis"] for feature in features] == [
        "checked_backlog_item",
        "linked_repository_evidence",
        "explicit_partial_marker",
        "open_without_linked_or_partial_evidence",
        "explicit_external_blocker",
    ]
    assert all(
        feature["qualification_status"] == "not_assessed_by_inventory" for feature in features
    )


def test_feature_map_rejects_checklist_items_outside_a_package() -> None:
    import pytest

    with pytest.raises(ValueError, match="outside a package"):
        build_feature_map("- [ ] orphan checklist item")


def test_feature_map_links_only_existing_repository_files(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/tool.py").write_text("pass\n")
    text = "### 0. Package\n- [ ] Use `scripts/tool.py` and `missing.py`.\n"
    result = build_feature_map(text, root=tmp_path)
    feature = result["features"][0]

    assert feature["evidence_links"]["cli_entries"] == ["scripts/tool.py"]
    assert feature["evidence_file_sha256"] == {
        "scripts/tool.py": "9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2"
    }
    assert feature["link_status"] == "linked"


def test_feature_map_resolves_unformatted_repository_paths(tmp_path: Path) -> None:
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/audit.py").write_text("pass\n")
    text = (
        "### 0. Package\n"
        "- [ ] Evidence is recorded in evaluation/audit.py; "
        "missing/path.json and an ordinary phrase are not evidence.\n"
    )

    feature = build_feature_map(text, root=tmp_path)["features"][0]

    assert feature["evidence_links"]["source_modules"] == ["evaluation/audit.py"]
    assert feature["evidence_file_sha256"] == {
        "evaluation/audit.py": "9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2"
    }
    assert feature["link_status"] == "linked"


def test_feature_map_resolves_dotted_modules_and_registered_cli(tmp_path: Path) -> None:
    integrations = tmp_path / "integrations"
    integrations.mkdir()
    (integrations / "runtime_inventory.py").write_text("def main(): pass\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\nloslab-runtime-inventory = "integrations.runtime_inventory:main"\n'
    )
    text = (
        "### 1. G0 runtime\n"
        "- [ ] Wire `integrations.runtime_inventory` into `loslab-runtime-inventory`.\n"
    )
    result = build_feature_map(text, root=tmp_path)
    feature = result["features"][0]

    assert feature["plan_gates"] == ["G0"]
    assert feature["evidence_links"]["source_modules"] == ["integrations/runtime_inventory.py"]
    assert feature["evidence_links"]["cli_entries"] == [
        "loslab-runtime-inventory -> integrations.runtime_inventory:main "
        "(integrations/runtime_inventory.py)"
    ]
    assert "integrations/runtime_inventory.py" in feature["evidence_file_sha256"]


def test_feature_map_rejects_symlinked_evidence(tmp_path: Path) -> None:
    import pytest

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/real.py").write_text("pass\n")
    (tmp_path / "scripts/alias.py").symlink_to("real.py")

    with pytest.raises(ValueError, match="symlink"):
        build_feature_map("### 0. Package\n- [ ] Use `scripts/alias.py`.\n", root=tmp_path)

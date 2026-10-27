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
    schema = json.loads(
        (root / "datasets/schema/implementation-feature-map.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)


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

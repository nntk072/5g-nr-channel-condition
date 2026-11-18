from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.build_historical_status_crosswalk import build_crosswalk


def test_crosswalk_indexes_explicit_ids_without_equating_generic_gate_names(tmp_path) -> None:
    (tmp_path / "docs/research").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_contract.py").write_text("pass", encoding="utf-8")
    ledger = {
        "requirements": [
            {
                "requirement_id": "PKG00-002",
                "backlog_aliases": ["PKG00-002", "PKG0.2"],
            },
            {"requirement_id": "DS.G4.07", "backlog_aliases": ["DS.G4.07"]},
        ]
    }
    (tmp_path / "docs/research/REQUIREMENT_LEDGER.json").write_text(
        json.dumps(ledger), encoding="utf-8"
    )
    (tmp_path / "docs/research/HISTORICAL_STATUS_MANUAL_MAPPINGS.json").write_text(
        json.dumps({"schema": "loslab.historical_status_manual_mappings.v1", "entries": []}),
        encoding="utf-8",
    )
    (tmp_path / "docs/research/MASTER_RESEARCH_PLAN.md").write_text(
        "\n".join(
            [
                "> **Current-state authority:** see ledger",
                "",
                "**Edition 1.2 G4 historical note (2026-10-01):** G4 remains open, while G4.07 is a criterion. See tests/test_contract.py.",
                "**Edition 1.3 implementation note (2026-10-02):** PKG00-002 maps to `tests/test_contract.py`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/research/MVP_STATUS.md").write_text(
        "**PKG00-002 review (2026-10-03):** Keep [the test](../../tests/test_contract.py).\n",
        encoding="utf-8",
    )
    schema_source = (
        Path(__file__).resolve().parents[2]
        / "docs/research/schema/historical-status-crosswalk.v1.schema.json"
    )
    schema_target = tmp_path / "docs/research/schema/historical-status-crosswalk.v1.schema.json"
    schema_target.parent.mkdir(parents=True)
    shutil.copyfile(schema_source, schema_target)
    manual_schema_source = (
        Path(__file__).resolve().parents[2]
        / "docs/research/schema/historical-status-manual-mappings.v1.schema.json"
    )
    manual_schema_target = (
        tmp_path / "docs/research/schema/historical-status-manual-mappings.v1.schema.json"
    )
    shutil.copyfile(manual_schema_source, manual_schema_target)

    payload = build_crosswalk(root=tmp_path)

    assert payload["entry_count"] == 3
    historical_gate, explicit_link, mvp_link = payload["entries"]
    assert historical_gate["heading"].startswith("Edition 1.2")
    assert historical_gate["direct_requirement_aliases"] == []
    assert historical_gate["requirement_ids"] == ["DS.G4.07"]
    assert historical_gate["numbered_dataset_criteria"] == ["G4.07"]
    assert historical_gate["mapping_state"] == "resolved_exact_match"
    assert historical_gate["cited_existing_paths"] == ["tests/test_contract.py"]
    assert explicit_link["direct_requirement_aliases"] == ["PKG00-002"]
    assert explicit_link["requirement_ids"] == ["PKG00-002"]
    assert explicit_link["cited_existing_paths"] == ["tests/test_contract.py"]
    assert mvp_link["requirement_ids"] == ["PKG00-002"]
    assert mvp_link["cited_existing_paths"] == ["tests/test_contract.py"]


def test_crosswalk_applies_heading_pinned_manual_mapping_with_rationale(tmp_path: Path) -> None:
    (tmp_path / "docs/research").mkdir(parents=True)
    ledger = {
        "requirements": [
            {"requirement_id": "DS.G4.07", "backlog_aliases": ["DS.G4.07"]},
        ]
    }
    (tmp_path / "docs/research/REQUIREMENT_LEDGER.json").write_text(
        json.dumps(ledger), encoding="utf-8"
    )
    heading = "Edition 2.1 generic G4 note (2026-10-01):"
    first_line = f"**{heading}** The retained evidence covers a finite angular confidence family."
    source_path = "docs/research/MASTER_RESEARCH_PLAN.md"
    (tmp_path / source_path).write_text(first_line + "\n", encoding="utf-8")
    (tmp_path / "docs/research/MVP_STATUS.md").write_text("", encoding="utf-8")
    entry_key = hashlib.sha256(f"{source_path}\0{1}\0{heading}".encode()).hexdigest()
    heading_sha256 = hashlib.sha256(heading.encode()).hexdigest()
    (tmp_path / "docs/research/HISTORICAL_STATUS_MANUAL_MAPPINGS.json").write_text(
        json.dumps(
            {
                "schema": "loslab.historical_status_manual_mappings.v1",
                "entries": [
                    {
                        "entry_key": entry_key,
                        "heading_sha256": heading_sha256,
                        "block_sha256": hashlib.sha256(first_line.encode()).hexdigest(),
                        "requirement_ids": ["DS.G4.07"],
                        "rationale": "The note states that it covers the finite G4.07 criterion family.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "historical-status-crosswalk.v1.schema.json",
        "historical-status-manual-mappings.v1.schema.json",
    ):
        schema_source = Path(__file__).resolve().parents[2] / "docs/research/schema" / name
        schema_target = tmp_path / "docs/research/schema" / name
        schema_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(schema_source, schema_target)

    payload = build_crosswalk(root=tmp_path)

    assert payload["resolved_manual_review_count"] == 1
    assert payload["needs_manual_mapping_count"] == 0
    assert payload["entries"][0]["mapping_state"] == "resolved_manual_review"
    assert payload["entries"][0]["requirement_ids"] == ["DS.G4.07"]
    assert payload["entries"][0]["manual_mapping"]["heading_sha256"] == heading_sha256

    (tmp_path / source_path).write_text(
        first_line + " Revised body changes the source note.\n", encoding="utf-8"
    )
    try:
        build_crosswalk(root=tmp_path)
    except ValueError as error:
        assert "note-block digest is stale" in str(error)
    else:
        raise AssertionError("changed source note body must invalidate its manual mapping")


def test_crosswalk_manual_mapping_fails_closed_on_unknown_requirement(tmp_path: Path) -> None:
    # Reuse the no-manual-map fixture and point a mapping at an unknown source key.
    (tmp_path / "docs/research").mkdir(parents=True)
    (tmp_path / "docs/research/REQUIREMENT_LEDGER.json").write_text(
        json.dumps({"requirements": [{"requirement_id": "DS.G4.07", "backlog_aliases": []}]}),
        encoding="utf-8",
    )
    (tmp_path / "docs/research/MASTER_RESEARCH_PLAN.md").write_text(
        "**Edition 1.1 note (2026-10-01):** generic gate note.\n", encoding="utf-8"
    )
    (tmp_path / "docs/research/MVP_STATUS.md").write_text("", encoding="utf-8")
    (tmp_path / "docs/research/HISTORICAL_STATUS_MANUAL_MAPPINGS.json").write_text(
        json.dumps(
            {
                "schema": "loslab.historical_status_manual_mappings.v1",
                "entries": [
                    {
                        "entry_key": "0" * 64,
                        "heading_sha256": "1" * 64,
                        "block_sha256": "2" * 64,
                        "requirement_ids": ["DS.DOES_NOT_EXIST"],
                        "rationale": "A sufficiently descriptive rationale for this mapping.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "historical-status-crosswalk.v1.schema.json",
        "historical-status-manual-mappings.v1.schema.json",
    ):
        schema_source = Path(__file__).resolve().parents[2] / "docs/research/schema" / name
        schema_target = tmp_path / "docs/research/schema" / name
        schema_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(schema_source, schema_target)

    try:
        build_crosswalk(root=tmp_path)
    except ValueError as error:
        assert "existing requirement IDs" in str(error)
    else:
        raise AssertionError("unknown manual requirement ID must fail closed")

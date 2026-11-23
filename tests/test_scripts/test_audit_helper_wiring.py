from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_helper_wiring import ARCHIVED_SOURCES, SCHEMA, build_inventory


def test_helper_wiring_inventory_binds_exports_and_direct_call_sites() -> None:
    inventory = build_inventory()
    schema = json.loads((Path(__file__).resolve().parents[2] / SCHEMA).read_text(encoding="utf-8"))
    from jsonschema import Draft202012Validator

    Draft202012Validator(schema).validate(inventory)
    assert inventory["helper_count"] == len(inventory["helpers"])
    assert inventory["helper_count"] >= 200
    assert "loslab" in inventory["reference_scan_roots"]
    assert inventory["test_reference_scan_roots"] == ["tests"]
    by_name = {item["qualified_name"]: item for item in inventory["helpers"]}

    phase = by_name["datasets.phase_comparison_disposition"]
    assert phase["documented_route"] is True
    assert any(
        item["path"] == "evaluation/complex_csi.py"
        for item in phase["direct_production_references"]
    )

    split_subject = by_name["datasets.standardized_draw_subject"]
    assert split_subject["documented_route"] is True
    assert split_subject["audit_disposition"] == "documented_route_requires_manual_review"
    assert split_subject["direct_production_references"]

    analytical_generator = by_name["datasets.generate_analytical_multipath"]
    assert analytical_generator["direct_production_references"] == []
    assert analytical_generator["exact_plan_reference_files"] == []
    assert analytical_generator["direct_test_references"]

    receiver_reader = by_name["datasets.read_receiver_model_inputs"]
    assert receiver_reader["direct_production_references"] == []
    assert receiver_reader["direct_test_references"]

    bundle_verifier = by_name["evaluation.verify_joint_evidence_bundle"]
    assert any(
        item["path"] == "tests/test_evaluation/test_joint_evaluation.py"
        for item in bundle_verifier["direct_test_references"]
    )

    planned_internal = {
        item["qualified_name"]: item for item in inventory["non_exported_plan_named_definitions"]
    }
    assert inventory["non_exported_plan_named_definition_count"] == len(planned_internal)
    assert "datasets.model_input_audit._validated_real_scalar" in planned_internal
    assert planned_internal["datasets.nr.from_receiver_output"]["definition_kind"] == "method"
    assert planned_internal["datasets.model_input_audit._validated_real_scalar"][
        "direct_production_references"
    ]
    assert planned_internal["receivers.nr.pusch_core.hopping.compose_pusch_intra_slot_hops"][
        "direct_test_references"
    ]

    for source, digest in inventory["source_sha256"].items():
        assert len(digest) == 64
        # Archived sources are read from the archive tag, not the working tree.
        assert (Path(__file__).resolve().parents[2] / source).is_file() or Path(
            source
        ) in ARCHIVED_SOURCES

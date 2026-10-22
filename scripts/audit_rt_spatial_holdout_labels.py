"""Recheck saved RT LOS/NLOS labels against independent scene visibility."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from datasets.channel import ChannelDataset
from integrations.sionna.los_oracle import SegmentVisibilityState, segment_visibility

LABEL_TO_VISIBILITY = {0: SegmentVisibilityState.BLOCKED, 1: SegmentVisibilityState.VISIBLE}
LABEL_NAMES = {0: "geometric_nlos", 1: "geometric_los"}


def compare_label_to_visibility(label: int, state: SegmentVisibilityState) -> bool:
    """Return whether a dataset's binary label matches the independent oracle."""
    if isinstance(label, bool) or label not in LABEL_TO_VISIBILITY:
        raise ValueError("geometric LOS/NLOS label must be integer 0 or 1")
    return state is LABEL_TO_VISIBILITY[label]


def _scene_tree_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene_xml.parent.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(scene_xml.parent).as_posix().encode("utf8"))
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def audit_spatial_holdout(dataset_path: Path, *, mitsuba_variant: str = "auto") -> dict[str, Any]:
    """Recompute finite-segment visibility for every saved sample and report mismatches."""
    dataset_path = dataset_path.resolve()
    dataset = ChannelDataset.load(dataset_path)
    if dataset.label_semantics != "geometric_direct_path_oracle":
        raise ValueError(
            "spatial label audit requires geometric_direct_path_oracle labels, "
            f"got {dataset.label_semantics!r}"
        )
    if len(dataset.metadata) != len(dataset.y):
        raise ValueError("dataset metadata and labels must have equal lengths")
    if not dataset.metadata:
        raise ValueError("dataset must contain at least one link")
    scene_paths = {Path(str(metadata["scene"])).resolve() for metadata in dataset.metadata}
    if len(scene_paths) != 1:
        raise ValueError("spatial holdout must contain one shared scene")
    scene_xml = scene_paths.pop()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"scene XML not found: {scene_xml}")

    from integrations.sionna.backend import activate_rt_backend, select_rt_backend

    backend_selection = select_rt_backend(mitsuba_variant)
    activate_rt_backend(backend_selection)
    import sionna.rt as rt

    scene = rt.load_scene(str(scene_xml))
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    labels = np.asarray(dataset.y).reshape(-1)
    for index, (raw_label, metadata) in enumerate(zip(labels, dataset.metadata, strict=True)):
        label = int(raw_label)
        if isinstance(raw_label, (bool, np.bool_)) or label not in LABEL_TO_VISIBILITY:
            raise ValueError(f"sample {index} has a non-binary geometric label {raw_label!r}")
        result = segment_visibility(scene, metadata["tx"], metadata["rx"])
        state = result.state
        recorded_name = LABEL_NAMES[label]
        oracle_name = {
            SegmentVisibilityState.VISIBLE: "geometric_los",
            SegmentVisibilityState.BLOCKED: "geometric_nlos",
        }.get(state, state.value.lower())
        matches = compare_label_to_visibility(label, state)
        counts[f"{recorded_name}__{oracle_name}"] += 1
        records.append(
            {
                "sample_index": index,
                "transmitter_id": metadata.get("transmitter_id"),
                "label": label,
                "recorded_label": recorded_name,
                "oracle_visibility": state.value,
                "oracle_label": oracle_name,
                "matches": matches,
                "tx_position_m": [float(value) for value in metadata["tx"]],
                "rx_position_m": [float(value) for value in metadata["rx"]],
                "segment_length_m": result.segment_length_m,
                "first_intersection_distance_m": result.first_intersection_distance_m,
                "incidence_cosine": result.incidence_cosine,
                "boundary_probe_blocked": list(result.boundary_probe_blocked),
            }
        )

    mismatch_count = sum(not row["matches"] for row in records)
    ambiguous_count = sum(row["oracle_visibility"] == "AMBIGUOUS" for row in records)
    return {
        "schema": "loslab.rt_spatial_holdout_label_audit.v1",
        "audit_passed": mismatch_count == 0 and ambiguous_count == 0,
        "label_contract": {
            "0": "geometric_nlos (blocked direct segment)",
            "1": "geometric_los (clear direct segment)",
            "source": "int(SegmentVisibilityResult.visible)",
        },
        "input": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "sample_count": len(records),
            "label_semantics": dataset.label_semantics,
        },
        "scene": {
            "xml_path": str(scene_xml),
            "xml_sha256": sha256_file(scene_xml),
            "scene_tree_sha256": _scene_tree_sha256(scene_xml),
        },
        "runtime": {
            "versions": versions(),
            "source_sha256": package_source_sha256(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "rt_backend_selection": backend_selection.as_dict(),
        },
        "counts": {
            "total": len(records),
            "mismatches": mismatch_count,
            "ambiguous": ambiguous_count,
            "recorded_vs_oracle": dict(sorted(counts.items())),
        },
        "records": records,
        "limitations": [
            "This audits geometric direct-segment labels only; it is not measured propagation truth.",
            "It does not validate solver convergence, material calibration, or channel coefficients.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="saved `nr-dmrs-observations.npz`")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mitsuba-variant", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing audit report: {args.output}")
    report = audit_spatial_holdout(args.dataset, mitsuba_variant=args.mitsuba_variant)
    atomic_json(args.output, report)
    print(
        f"Audited {report['counts']['total']} links: "
        f"{report['counts']['mismatches']} mismatches, "
        f"{report['counts']['ambiguous']} ambiguous; "
        f"passed={report['audit_passed']}"
    )
    if not report["audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

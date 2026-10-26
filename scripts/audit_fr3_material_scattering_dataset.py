"""Audit the open 8–12 GHz building-material scattering measurement record.

This validates source-file integrity and Touchstone structure only. It does not
fit a Sionna material, transfer parameters to 3.5 GHz, or qualify G3.03.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_ID = "15610052"
DOI = "10.5281/zenodo.15610052"
LICENSE = "CC-BY-4.0"
ARCHIVE_NAME = "Scattering Measurements Materials FR3 CNIT.zip"
PDF_NAME = "Info Dataset Scattering Measurements FR3 - CNIT.pdf"
EXPECTED_ARCHIVE_MD5 = "a9886323b7831d0ee49ca53b4f630594"
EXPECTED_PDF_MD5 = "cb8efdf3b623c1b8c0c2fe7ba5dc1d37"
EXPECTED_ARCHIVE_SHA256 = "fc7efdf50cbbb050d8ceee1c8bce551bdcd05ed35fc27c107aabca9b6bff2ac4"
EXPECTED_PDF_SHA256 = "ede144a2f92368bb052a1ee7e0e007447fa030dde021441e2cd7fdec7b2fafa0"
MATERIALS = (
    "Brick Wall",
    "Gypsum Board Only",
    "Marble",
    "Plasterbord With Stud Metal Inside",
    "PlyWood",
)
RX_ANGLES_DEG = tuple(range(10, 171, 10))
FREQUENCY_BOUNDS_HZ = (8e9, 12e9)
FREQUENCY_POINTS = 1601
_MEASUREMENT_NAME = re.compile(r"_(?:RX)(\d+)_TX135\.s2p$", re.IGNORECASE)


def _digest(payload: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, payload).hexdigest()


def _parse_s2p(data: bytes, member: str) -> tuple[float, ...]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"Touchstone member is not ASCII: {member}") from error
    option_line: str | None = None
    frequencies: list[float] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        content = line.split("!", maxsplit=1)[0].strip()
        if not content:
            continue
        if content.startswith("#"):
            if option_line is not None:
                raise ValueError(f"multiple Touchstone option lines in {member}")
            option_line = " ".join(content[1:].upper().split())
            continue
        values = content.split()
        if len(values) != 9:
            raise ValueError(f"expected 9 Touchstone S2P columns at {member}:{line_number}")
        try:
            row = [float(value) for value in values]
        except ValueError as error:
            raise ValueError(f"nonnumeric Touchstone row at {member}:{line_number}") from error
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"nonfinite Touchstone row at {member}:{line_number}")
        frequencies.append(row[0])
    if option_line != "HZ S DB R 50":
        raise ValueError(f"unexpected Touchstone format in {member}: {option_line!r}")
    if len(frequencies) != FREQUENCY_POINTS:
        raise ValueError(f"unexpected frequency row count in {member}: {len(frequencies)}")
    if frequencies[0] != FREQUENCY_BOUNDS_HZ[0] or frequencies[-1] != FREQUENCY_BOUNDS_HZ[1]:
        raise ValueError(f"unexpected frequency bounds in {member}")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError(f"frequency axis is not strictly increasing in {member}")
    expected_step = (FREQUENCY_BOUNDS_HZ[1] - FREQUENCY_BOUNDS_HZ[0]) / (FREQUENCY_POINTS - 1)
    if any(
        not math.isclose(right - left, expected_step, rel_tol=0.0, abs_tol=1e-6)
        for left, right in zip(frequencies, frequencies[1:])
    ):
        raise ValueError(f"frequency axis is not uniformly sampled in {member}")
    return tuple(frequencies)


def inspect_touchstone_archive(archive_bytes: bytes) -> dict[str, Any]:
    """Validate the expected five-material, 17-angle Touchstone bundle."""
    if len(archive_bytes) > 32 * 1024 * 1024:
        raise ValueError("archive exceeds the 32 MiB input limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as error:
        raise ValueError("dataset file is not a valid ZIP archive") from error
    with archive:
        entries = archive.infolist()
        if sum(info.file_size for info in entries) > 32 * 1024 * 1024:
            raise ValueError("archive expands beyond the 32 MiB limit")
        if len(entries) > 256:
            raise ValueError("archive contains too many ZIP members")
        measured_paths: dict[str, dict[int, str]] = defaultdict(dict)
        control_paths: dict[str, list[str]] = defaultdict(list)
        frequency_axis: tuple[float, ...] | None = None
        touchstone_count = 0
        for info in entries:
            path = Path(info.filename)
            if (
                path.is_absolute()
                or "\\" in info.filename
                or re.match(r"^[A-Za-z]:", info.filename) is not None
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(f"unsafe ZIP member path: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"ZIP symlink member is not allowed: {info.filename}")
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(".s2p"):
                raise ValueError(f"unexpected non-Touchstone archive file: {info.filename}")
            if info.file_size > 512 * 1024:
                raise ValueError(f"oversized Touchstone member: {info.filename}")
            parts = info.filename.split("/")
            if len(parts) == 3 and parts[0] == "Scattering Measurements Materials FR3 CNIT":
                folder_index, filename_index = 1, 2
            elif len(parts) == 2:
                folder_index, filename_index = 0, 1
            else:
                raise ValueError(f"unexpected Touchstone directory layout: {info.filename}")
            if not parts[folder_index].startswith("Scattering Measurements "):
                raise ValueError(f"unexpected Touchstone directory layout: {info.filename}")
            material = (
                parts[folder_index]
                .removeprefix("Scattering Measurements ")
                .removesuffix(" CNIT Tx located 135 degree")
            )
            if material not in MATERIALS:
                raise ValueError(f"unknown material folder: {parts[folder_index]}")
            data = archive.read(info)
            current_axis = _parse_s2p(data, info.filename)
            if frequency_axis is None:
                frequency_axis = current_axis
            elif current_axis != frequency_axis:
                raise ValueError(f"frequency grid differs across files: {info.filename}")
            touchstone_count += 1
            filename = parts[filename_index]
            match = _MEASUREMENT_NAME.search(filename)
            if match is not None:
                angle = int(match.group(1))
                if angle in measured_paths[material]:
                    raise ValueError(f"duplicate RX angle {angle} for {material}")
                measured_paths[material][angle] = info.filename
            else:
                lowered = filename.lower()
                if "calibration" in lowered or "freespace" in lowered or "free_space" in lowered:
                    control_paths[material].append(info.filename)
                else:
                    raise ValueError(f"unclassified Touchstone file: {info.filename}")

        if touchstone_count != 95:
            raise ValueError(f"expected 95 Touchstone files, found {touchstone_count}")
        for material in MATERIALS:
            if set(measured_paths[material]) != set(RX_ANGLES_DEG):
                raise ValueError(f"{material} does not contain the expected 17 RX angles")
            if len(control_paths[material]) != 2:
                raise ValueError(f"{material} does not contain two calibration/free-space controls")
        if frequency_axis is None:
            raise ValueError("archive contains no Touchstone S2P files")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
    return {
        "touchstone_file_count": touchstone_count,
        "measured_profile_count": sum(len(measured_paths[item]) for item in MATERIALS),
        "reference_file_count": sum(len(control_paths[item]) for item in MATERIALS),
        "material_profile_counts": {item: len(measured_paths[item]) for item in MATERIALS},
        "rx_angles_deg": list(RX_ANGLES_DEG),
        "frequency_start_hz": frequency_axis[0],
        "frequency_end_hz": frequency_axis[-1],
        "frequency_step_hz": frequency_axis[1] - frequency_axis[0],
        "frequency_points_per_file": len(frequency_axis),
        "frequency_grid_sha256": hashlib.sha256(
            json.dumps(frequency_axis, separators=(",", ":")).encode()
        ).hexdigest(),
        "zip_crc": "passed",
    }


def audit_dataset(data_dir: Path) -> dict[str, Any]:
    archive_path = data_dir / ARCHIVE_NAME
    pdf_path = data_dir / PDF_NAME
    archive_bytes = archive_path.read_bytes()
    pdf_bytes = pdf_path.read_bytes()
    actual = {
        "archive_md5": _digest(archive_bytes, "md5"),
        "archive_sha256": _digest(archive_bytes, "sha256"),
        "pdf_md5": _digest(pdf_bytes, "md5"),
        "pdf_sha256": _digest(pdf_bytes, "sha256"),
    }
    expected = {
        "archive_md5": EXPECTED_ARCHIVE_MD5,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "pdf_md5": EXPECTED_PDF_MD5,
        "pdf_sha256": EXPECTED_PDF_SHA256,
    }
    if actual != expected:
        raise ValueError("downloaded Zenodo files differ from the audited record version")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core = {
        "schema": "umcd.external_material_scattering_source_audit.v1",
        "status": "OUT_OF_BAND_REFERENCE_ONLY",
        "source": {
            "zenodo_record_id": DATASET_ID,
            "doi": DOI,
            "title": "Scattering Measurements of Building Materials in the FR3 band",
            "authors": ["Elena Bernardi", "Enrico Maria Vitucci", "Vittorio Degli-Esposti"],
            "license": LICENSE,
            "license_source": "Zenodo record metadata API; license id cc-by-4.0",
            "record_access": "open",
            "retrieved_utc_date": "2026-10-24",
            "files": {
                ARCHIVE_NAME: {
                    "bytes": len(archive_bytes),
                    "md5": actual["archive_md5"],
                    "sha256": actual["archive_sha256"],
                },
                PDF_NAME: {
                    "bytes": len(pdf_bytes),
                    "md5": actual["pdf_md5"],
                    "sha256": actual["pdf_sha256"],
                },
            },
        },
        "measurement_audit": inspect_touchstone_archive(archive_bytes),
        "setup_from_source_pdf": {
            "transmitter_angle_deg": 135,
            "receiver_angles_deg": list(RX_ANGLES_DEG),
            "receiver_step_deg": 10,
            "geometry": "horizontal bistatic semicircle around the sample; planar angular sweep",
            "tx_wall_distance_m_by_material": {
                "Plasterbord With Stud Metal Inside": 1.60,
                "Brick Wall": 1.60,
                "Gypsum Board Only": 1.60,
                "PlyWood": 0.32,
                "Marble": 0.32,
            },
            "rx_wall_distance_m_by_material": {
                "Plasterbord With Stud Metal Inside": 2.45,
                "Brick Wall": 2.45,
                "Gypsum Board Only": 2.45,
                "PlyWood": 1.85,
                "Marble": 1.73,
            },
            "height_m_by_material": {
                "Plasterbord With Stud Metal Inside": 0.60,
                "Brick Wall": 0.43,
                "Gypsum Board Only": 0.60,
                "PlyWood": 0.78,
                "Marble": 0.78,
            },
            "controls_per_material": [
                "same-day SOLT thru/cable measurement",
                "free-space Tx/Rx reference",
            ],
        },
        "evidence_boundary": {
            "supports": [
                "repeatable parsing and frequency-grid checks for measured 8–12 GHz S-parameters",
                "planar 10–170 degree receiver-angle observations for five named sample constructions",
                "a separately scoped out-of-band angular-scattering model check after calibration review",
            ],
            "does_not_support": [
                "3.5 GHz target-frequency calibration or frequency transfer without independent validation",
                "full-sphere 3D angular scattering",
                "independent material/acquisition-unit uncertainty estimates from the single angular sweep",
                "direct mapping from S21 to Sionna's diffuse coefficient without antenna, geometry, and power normalization",
                "complex path-level CSI or NR receiver qualification",
            ],
            "qualification": "not_claimed",
            "next_scientific_step": "derive a preregistered power/angular observable from the documented calibration and geometry, then validate it on held-out angles/frequencies; keep target-band qualification separate",
        },
        "audit_source_sha256": source_sha256,
    }
    core["report_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/external/fr3-building-scattering-zenodo-15610052"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/results/g3-fr3-building-scattering-audit-20261024.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing audit: {args.output}")
    report = audit_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report_sha256": report["report_sha256"]}))


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import zipfile

import pytest
from scripts.audit_fr3_material_scattering_dataset import (
    FREQUENCY_BOUNDS_HZ,
    FREQUENCY_POINTS,
    MATERIALS,
    RX_ANGLES_DEG,
    inspect_touchstone_archive,
)


def _touchstone_payload(
    *, corrupt_last_frequency: bool = False, omit_one_profile: bool = False
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for material in MATERIALS:
            folder = f"Scattering Measurements {material} CNIT Tx located 135 degree"
            filenames = [
                f"{material.replace(' ', '')}_RX{angle}_TX135.s2p" for angle in RX_ANGLES_DEG
            ]
            if omit_one_profile and material == MATERIALS[-1]:
                filenames.remove(f"{material.replace(' ', '')}_RX10_TX135.s2p")
            filenames.extend(("CalibrationMeas_fixture.s2p", "FreeSpace_fixture.s2p"))
            for filename in filenames:
                rows = ["# Hz S dB R 50"]
                for index in range(FREQUENCY_POINTS):
                    frequency = FREQUENCY_BOUNDS_HZ[0] + index * (
                        FREQUENCY_BOUNDS_HZ[1] - FREQUENCY_BOUNDS_HZ[0]
                    ) / (FREQUENCY_POINTS - 1)
                    if (
                        corrupt_last_frequency
                        and material == MATERIALS[-1]
                        and filename == filenames[0]
                        and index == FREQUENCY_POINTS - 1
                    ):
                        frequency -= 1e6
                    rows.append(f"{frequency:.1f} 0 0 0 0 0 0 0 0")
                archive.writestr(
                    f"Scattering Measurements Materials FR3 CNIT/{folder}/{filename}",
                    "\n".join(rows) + "\n",
                )
    return output.getvalue()


def test_touchstone_archive_validates_all_materials_angles_and_common_frequency_grid() -> None:
    result = inspect_touchstone_archive(_touchstone_payload())

    assert result["touchstone_file_count"] == 95
    assert result["measured_profile_count"] == 85
    assert result["reference_file_count"] == 10
    assert result["rx_angles_deg"] == list(RX_ANGLES_DEG)
    assert result["frequency_points_per_file"] == 1601
    assert result["frequency_start_hz"] == 8e9
    assert result["frequency_end_hz"] == 12e9
    assert result["zip_crc"] == "passed"


def test_touchstone_archive_rejects_invalid_zip() -> None:
    with pytest.raises(ValueError, match="valid ZIP archive"):
        inspect_touchstone_archive(b"not a zip")


def test_touchstone_archive_rejects_omitted_measurement_profile() -> None:
    with pytest.raises(ValueError, match="expected 95 Touchstone files"):
        inspect_touchstone_archive(_touchstone_payload(omit_one_profile=True))


def test_touchstone_archive_rejects_one_file_with_different_frequency_axis() -> None:
    with pytest.raises(ValueError, match="unexpected frequency bounds"):
        inspect_touchstone_archive(_touchstone_payload(corrupt_last_frequency=True))

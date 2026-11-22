from pathlib import Path

from agentops.doctor import doctor_report

ROOT = Path(__file__).resolve().parents[2]


def test_doctor_checks_toolchain_without_native_imports() -> None:
    report = doctor_report(ROOT)
    assert report["ok"] is True
    assert report["uv"] != "UNAVAILABLE"
    assert report["native_import_performed"] is False
    assert report["scientific_qualification"] == "UNCHANGED"
    assert all(value == "AVAILABLE" for value in report["package_imports"].values())
    assert all(
        item["ignored"] and item["writable_parent"] for item in report["state_roots"].values()
    )

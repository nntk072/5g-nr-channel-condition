"""Bounded, pickle-free audit of official CAEZ numeric archives."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from datasets import caez


def _archive(
    path,
    *,
    complex_h: bool = True,
    samples: int = 12,
    timestamps=None,
    positions=None,
    extra=None,
    compressed: bool = False,
):
    rng = np.random.default_rng(4)
    h = rng.normal(size=(samples, 4)) + (1j * rng.normal(size=(samples, 4)) if complex_h else 0)
    fields = {
        "H": h if complex_h else h.real,
        "noise_var": np.full(samples, 0.01),
        "timestamps": np.arange(samples, dtype=np.float64) * 0.5
        if timestamps is None
        else timestamps,
        "pos": rng.normal(size=(samples, 3)) if positions is None else positions,
    }
    fields.update(extra or {})
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **fields)
    return path


def test_complex_archive_reports_fields_order_and_the_phase_disposition(tmp_path) -> None:
    path = _archive(tmp_path / "caez.npz")
    report = caez.audit(path, chunk_rows=5)
    assert report["schema"] == "caez-preprocessed-npz-audit.v1"
    assert report["zip_crc_check"] == "passed"
    assert report["sample_count"] == 12
    assert report["fields"]["H"]["all_finite"] is True
    assert report["fields"]["H"]["shape"] == [12, 4]
    assert report["channel_observation"]["complex_coefficients_present"] is True
    assert (
        report["channel_observation"]["g3_complex_csi_disposition"]
        == "not_established_phase_reference_unverified"
    )
    assert report["timestamps"]["strictly_increasing"] is True
    assert report["timestamps"]["min_positive_step"] == pytest.approx(0.5)
    assert report["timestamps"]["max_step"] == pytest.approx(0.5)
    assert report["archive_sha256"] == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert "R. Wiesmayr" in report["required_citation"]
    assert report["sample_split_policy"].startswith("chronological contiguous")


def test_real_archive_is_ineligible_for_complex_csi(tmp_path) -> None:
    report = caez.audit(_archive(tmp_path / "real.npz", complex_h=False))
    assert report["channel_observation"]["complex_coefficients_present"] is False
    assert (
        report["channel_observation"]["g3_complex_csi_disposition"]
        == "ineligible_no_explicit_complex_phase"
    )


def test_non_monotonic_and_single_step_timestamps_are_reported(tmp_path) -> None:
    unsorted = caez.audit(
        _archive(tmp_path / "unsorted.npz", timestamps=np.array([0.0, 2.0, 1.0] + [3.0] * 9))
    )
    assert unsorted["timestamps"]["strictly_increasing"] is False
    constant = caez.audit(_archive(tmp_path / "constant.npz", timestamps=np.zeros(12)))
    assert constant["timestamps"]["min_positive_step"] is None
    assert constant["timestamps"]["max_step"] == pytest.approx(0.0)


def test_nonfinite_channel_values_are_counted_not_hidden(tmp_path) -> None:
    rng = np.random.default_rng(2)
    h = rng.normal(size=(12, 4)).astype(np.complex128)
    h[3, 1] = complex(np.nan, 0.0)
    path = tmp_path / "nan.npz"
    np.savez(
        path, H=h, noise_var=np.ones(12), timestamps=np.arange(12.0), pos=rng.normal(size=(12, 2))
    )
    report = caez.audit(path)
    assert report["fields"]["H"]["all_finite"] is False
    assert report["fields"]["H"]["finite_values"] == report["fields"]["H"]["total_values"] - 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extra": {"noise_var": None}}, "NPY payload|must contain|numeric"),
        ({"positions": np.zeros((12, 1))}, "at least two coordinates"),
        ({"timestamps": np.zeros((12, 2))}, "one-dimensional real numeric"),
        ({"timestamps": np.array(["a"] * 12)}, "numeric NPY array"),
        ({"samples": 0}, "non-empty sample axis"),
        ({"compressed": True}, "ZIP_STORED"),
    ],
)
def test_archive_shape_and_storage_rules(tmp_path, kwargs, message) -> None:
    kwargs = dict(kwargs)
    if kwargs.get("extra", {}).get("noise_var", 1) is None:
        kwargs["extra"] = {"noise_var": np.array([object()] * 12, dtype=object)}
    if kwargs.get("samples") == 0:
        path = tmp_path / "empty.npz"
        np.savez(
            path,
            H=np.zeros((0, 4), dtype=np.complex128),
            noise_var=np.zeros(0),
            timestamps=np.zeros(0),
            pos=np.zeros((0, 2)),
        )
    else:
        path = _archive(tmp_path / "bad.npz", **kwargs)
    with pytest.raises(ValueError, match=message):
        caez.audit(path)


def test_missing_and_mismatched_fields_are_rejected(tmp_path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(
        path, H=np.ones((4, 4), dtype=complex), noise_var=np.ones(4), timestamps=np.arange(4.0)
    )
    with pytest.raises(ValueError, match="required CAEZ array is missing: pos"):
        caez.audit(path)
    mismatched = _archive(tmp_path / "mismatch.npz", extra={"noise_var": np.ones(5)})
    with pytest.raises(ValueError, match="do not share a sample count"):
        caez.audit(mismatched)


def test_non_archives_and_invalid_chunking_are_rejected(tmp_path) -> None:
    text = tmp_path / "plain.txt"
    text.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid NPZ/ZIP"):
        caez.audit(text)
    path = _archive(tmp_path / "ok.npz")
    with pytest.raises(ValueError, match="chunk_rows must be positive"):
        caez.audit(path, chunk_rows=0)


def test_corrupted_member_fails_the_zip_crc_check(tmp_path) -> None:
    path = _archive(tmp_path / "corrupt.npz")
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("H.npy")
    data = bytearray(path.read_bytes())
    offset = info.header_offset + 30 + len(info.filename.encode()) + 200
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="CRC check failed|must|invalid|truncated|NPY"):
        caez.audit(path)


def test_cli_prints_the_report_and_can_write_it(tmp_path, monkeypatch, capsys) -> None:
    path = _archive(tmp_path / "cli.npz")
    monkeypatch.setattr("sys.argv", ["caez", str(path)])
    assert caez.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sample_count"] == 12
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(
        "sys.argv", ["caez", str(path), "--chunk-rows", "4", "--output", str(output)]
    )
    assert caez.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["fields"]["H"]["shape"] == [12, 4]

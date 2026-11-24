"""Profile-proxy dataset runner: overwrite refusal, provenance manifest and CLI error mapping."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets.channel import ChannelDataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location(
        "profile_proxy_runner_under_test", ROOT / "scripts" / "run_profile_proxy_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset() -> ChannelDataset:
    return ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["cdl", "tdl"]),
        source=np.asarray(["cdl", "tdl"]),
        metadata=(
            {"family": "cdl", "profile": "A", "spread_ns": 30.0},
            {"family": "tdl", "profile": "E", "spread_ns": 100.0},
        ),
    )


def _kwargs(**overrides):
    values = {
        "samples_per_profile": 1,
        "seed": 7,
        "device": "cpu",
        "spec_version": "19.2",
        "snr_db": 20.0,
        "n_subcarriers": 8,
        "delay_spreads_ns": [30.0, 100.0],
    }
    values.update(overrides)
    return values


def test_existing_output_or_manifest_is_never_overwritten(runner, tmp_path: Path) -> None:
    output = tmp_path / "profiles.npz"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_dataset(output, **_kwargs())

    manifest_only = tmp_path / "other.npz"
    (tmp_path / "other.npz.manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_dataset(manifest_only, **_kwargs())


@pytest.mark.parametrize("snr", [float("nan"), float("inf")])
def test_non_finite_snr_is_rejected_before_generation(runner, tmp_path: Path, snr) -> None:
    with pytest.raises(ValueError, match="snr_db must be finite"):
        runner.run_dataset(tmp_path / "profiles.npz", **_kwargs(snr_db=snr))
    assert not (tmp_path / "profiles.npz").exists()


def test_successful_run_writes_dataset_and_labelled_manifest(
    runner, tmp_path: Path, monkeypatch
) -> None:
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return _dataset()

    monkeypatch.setattr(runner, "generate_standardized", fake_generate)
    monkeypatch.setattr(runner, "versions", lambda: {"git_sha": "abc123", "numpy": "x"})
    output = tmp_path / "profiles.npz"

    manifest = runner.run_dataset(output, **_kwargs())

    assert calls["profiles"] == ("A", "B", "C", "D", "E")
    assert calls["families"] == ("tdl", "cdl")
    assert calls["per_profile"] == 1
    assert calls["delay_spreads_ns"] == [30.0, 100.0]
    assert output.exists()
    on_disk = json.loads((tmp_path / "profiles.npz.manifest.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(manifest))
    assert manifest["schema"] == "loslab.profile_proxy_dataset.v1"
    assert manifest["evidence_grade"] == "SIMULATED_3GPP_PROFILE_PROXY_NOT_GEOMETRIC_LOS"
    assert manifest["source_revision"] == "abc123"
    assert manifest["sample_count"] == 2
    assert manifest["class_counts"] == {"0": 1, "1": 1}
    assert manifest["configuration"]["carrier_hz"] > 0
    assert manifest["configuration"]["n_subcarriers"] == 8
    assert any("not geometric" in text for text in manifest["limitations"])


def test_hardware_block_reports_cuda_only_when_available(
    runner, tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(runner, "generate_standardized", lambda **_: _dataset())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Fake GPU")

    manifest = runner.run_dataset(tmp_path / "gpu.npz", **_kwargs(device="cuda:0"))

    assert manifest["hardware"] == {
        "device": "cuda:0",
        "cuda_available": True,
        "cuda_device_name": "Fake GPU",
    }


def test_missing_torch_is_recorded_as_no_cuda(runner, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "generate_standardized", lambda **_: _dataset())
    monkeypatch.setitem(sys.modules, "torch", None)

    manifest = runner.run_dataset(tmp_path / "cpu.npz", **_kwargs(device="cuda:0"))

    assert manifest["hardware"] == {"device": "cuda:0", "cuda_available": False}


def test_cli_reports_runtime_errors_as_parser_errors(
    runner, tmp_path: Path, monkeypatch, capsys
) -> None:
    def failing(*_args, **_kwargs):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(runner, "run_dataset", failing)
    monkeypatch.setattr(sys, "argv", ["run_profile_proxy_dataset.py", str(tmp_path / "x.npz")])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 2
    assert "CUDA unavailable" in capsys.readouterr().err


def test_cli_prints_saved_row_count_and_manifest_path(
    runner, tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "cli.npz"
    monkeypatch.setattr(runner, "run_dataset", lambda *_args, **_kwargs: {"sample_count": 10})
    monkeypatch.setattr(sys, "argv", ["run_profile_proxy_dataset.py", str(output)])

    runner.main()

    out = capsys.readouterr().out
    assert "Saved 10 profile-proxy rows" in out
    assert str(output.with_suffix(".npz.manifest.json")) in out

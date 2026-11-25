from __future__ import annotations

import json

import pytest
from scripts.generate_rt_dataset import main


def test_rt_data_cli_defaults_to_native_capability_detection(monkeypatch, capsys) -> None:
    import integrations.sionna.adapter as sionna2

    captured = {}

    class FakeDataset:
        y = [0]

        def save(self, path):
            captured["output"] = path

    def fake_generate(config, *, seed, **_kwargs):
        captured["config"] = config
        captured["seed"] = seed
        return FakeDataset()

    monkeypatch.setattr(sionna2, "generate_raytraced", fake_generate)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("sys.argv", ["generate_rt_dataset", "--pilot", "--output", "unused.npz"])

    main()

    assert captured["config"].mitsuba_variant == "auto"
    assert captured["seed"] == 42
    assert "GEOMETRY-LABELED RT" in capsys.readouterr().out


def test_rt_data_cli_persists_generation_failure_report_without_writing_dataset(
    tmp_path, monkeypatch, capsys
) -> None:
    import integrations.sionna.adapter as sionna2

    output_path = tmp_path / "failed.npz"
    failure_path = tmp_path / "failed.npz.failure.json"
    report = {
        "status": "insufficient_samples",
        "attempt_counts": {"candidate_geometric_los": 20, "candidate_geometric_nlos": 0},
    }

    def fail_generation(config, *, seed, **_kwargs):
        raise sionna2.RaytraceGenerationError("No NLOS samples", report)

    monkeypatch.setattr(sionna2, "generate_raytraced", fail_generation)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "sys.argv", ["generate_rt_dataset", "--pilot", "--output", str(output_path)]
    )

    with pytest.raises(sionna2.RaytraceGenerationError, match="No NLOS samples"):
        main()

    assert not output_path.exists()
    assert json.loads(failure_path.read_text(encoding="utf-8")) == report
    assert str(failure_path) in capsys.readouterr().err


def test_rt_data_cli_does_not_overwrite_existing_failure_report(monkeypatch, tmp_path) -> None:
    import integrations.sionna.adapter as sionna2

    output_path = tmp_path / "failed.npz"
    failure_path = tmp_path / "failed.npz.failure.json"
    failure_path.write_text("preserve", encoding="utf-8")

    def fail_generation(config, *, seed, **_kwargs):
        raise sionna2.RaytraceGenerationError("No NLOS samples", {"status": "failed"})

    monkeypatch.setattr(sionna2, "generate_raytraced", fail_generation)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "sys.argv", ["generate_rt_dataset", "--pilot", "--output", str(output_path)]
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()

    assert failure_path.read_text(encoding="utf-8") == "preserve"


def test_rt_data_cli_dispatches_to_content_addressed_cache_when_requested(
    monkeypatch, capsys, tmp_path
) -> None:
    import integrations.sionna.adapter as sionna2

    captured = {}

    class FakeDataset:
        y = [0]

        def save(self, path):
            captured["output"] = path

    def unexpected_uncached(*args, **kwargs):
        raise AssertionError("cache option should use the cache-aware RT entry point")

    def cached(config, *, cache_directory, seed, **_kwargs):
        captured.update({"config": config, "cache_directory": cache_directory, "seed": seed})
        return FakeDataset()

    monkeypatch.setattr(sionna2, "generate_raytraced", unexpected_uncached)
    monkeypatch.setattr(sionna2, "generate_raytraced_cached", cached)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_rt_dataset",
            "--pilot",
            "--output",
            "unused.npz",
            "--cache-dir",
            str(tmp_path / "rt-cache"),
        ],
    )

    main()

    assert captured["cache_directory"] == str(tmp_path / "rt-cache")
    assert captured["config"].mitsuba_variant == "auto"
    assert captured["seed"] == 42
    assert "GEOMETRY-LABELED RT" in capsys.readouterr().out


def test_rt_data_cli_rejects_empty_production_approval_digest(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_rt_dataset",
            "--output",
            str(tmp_path / "production.npz"),
            "--qualification-approval",
            str(tmp_path / "approval.json"),
            "--qualification-approval-sha256",
            "",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "--qualification-approval-sha256 is required with production approval" in capsys.readouterr().err


def test_rt_data_cli_reports_failure_to_persist_generation_diagnostic(
    monkeypatch, tmp_path, capsys
) -> None:
    import artifacts.provenance as provenance
    import integrations.sionna.adapter as sionna2

    output = tmp_path / "failed.npz"
    failure_path = tmp_path / "failed.npz.failure.json"

    def fail_generation(_config, *, seed, **_kwargs):
        raise sionna2.RaytraceGenerationError(
            "No NLOS samples", {"status": "insufficient_samples"}
        )

    monkeypatch.setattr(sionna2, "generate_raytraced", fail_generation)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        provenance,
        "atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    monkeypatch.setattr("sys.argv", ["generate_rt_dataset", "--pilot", "--output", str(output)])

    with pytest.raises(sionna2.RaytraceGenerationError, match="No NLOS samples"):
        main()

    assert not failure_path.exists()
    assert "attempt report could not be saved" in capsys.readouterr().err


def test_rt_data_cli_routes_authorization_errors_to_argument_parser(monkeypatch, tmp_path, capsys):
    import integrations.sionna.adapter as sionna2

    monkeypatch.setattr(
        sionna2,
        "authorize_rt_data_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid pilot request")),
    )
    monkeypatch.setattr("sys.argv", ["generate_rt_dataset", "--pilot", "--output", str(tmp_path / "out.npz")])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "invalid pilot request" in capsys.readouterr().err

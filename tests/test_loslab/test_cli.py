from __future__ import annotations

import json

import pytest
from datasets.channel import ChannelDataset
from loslab.__main__ import main


def test_doctor_reports_environment_without_running_simulation(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["loslab", "doctor"])

    main()

    report = json.loads(capsys.readouterr().out)
    assert "python" in report
    assert "sionna" in report
    assert "rt_backend" not in report


def test_doctor_can_probe_and_report_resolved_rt_backend(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    import integrations.sionna.backend as backend

    monkeypatch.setattr("loslab.provenance.versions", lambda: {"python": "3.12.0"})
    monkeypatch.setattr(
        backend,
        "select_rt_backend",
        lambda requested: SimpleNamespace(
            as_dict=lambda: {
                "requested": requested,
                "backend": "llvm_cpu",
                "variant": "llvm_ad_mono_polarized",
                "probe_passed": True,
            }
        ),
    )
    monkeypatch.setattr("sys.argv", ["loslab", "doctor", "--probe-rt", "--mitsuba-variant", "auto"])

    main()

    report = json.loads(capsys.readouterr().out)
    assert report["rt_backend"]["probe_passed"] is True
    assert report["rt_backend"]["backend"] == "llvm_cpu"
    assert report["rt_backend"]["requested"] == "auto"


def test_doctor_reports_unavailable_rt_backend_as_json(monkeypatch, capsys) -> None:
    import integrations.sionna.backend as backend

    monkeypatch.setattr("loslab.provenance.versions", lambda: {"python": "3.12.0"})

    def fail_selection(requested):
        raise RuntimeError("no usable Sionna RT backend")

    monkeypatch.setattr(backend, "select_rt_backend", fail_selection)
    monkeypatch.setattr("sys.argv", ["loslab", "doctor", "--probe-rt"])

    main()

    report = json.loads(capsys.readouterr().out)
    assert report["rt_backend"]["probe_passed"] is False
    assert "no usable" in report["rt_backend"]["error"]


def test_debug_data_command_writes_loadable_dataset(tmp_path, monkeypatch, capsys) -> None:
    destination = tmp_path / "debug.npz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "loslab",
            "debug-data",
            "--output",
            str(destination),
            "--scenes",
            "2",
            "--links-per-scene",
            "4",
        ],
    )

    main()

    assert isinstance(ChannelDataset.load(destination), ChannelDataset)
    assert "DEBUG DATA" in capsys.readouterr().out


def test_rt_data_cli_defaults_to_native_capability_detection(monkeypatch, capsys) -> None:
    import loslab.sionna2 as sionna2

    captured = {}

    class FakeDataset:
        y = [0]

        def save(self, path):
            captured["output"] = path

    def fake_generate(config, *, seed):
        captured["config"] = config
        captured["seed"] = seed
        return FakeDataset()

    monkeypatch.setattr(sionna2, "generate_raytraced", fake_generate)
    monkeypatch.setattr("sys.argv", ["loslab", "rt-data", "--output", "unused.npz"])

    main()

    assert captured["config"].mitsuba_variant == "auto"
    assert captured["seed"] == 42
    assert "GEOMETRY-LABELED RT" in capsys.readouterr().out


def test_rt_data_cli_dispatches_to_content_addressed_cache_when_requested(
    monkeypatch, capsys, tmp_path
) -> None:
    import loslab.sionna2 as sionna2

    captured = {}

    class FakeDataset:
        y = [0]

        def save(self, path):
            captured["output"] = path

    def unexpected_uncached(*args, **kwargs):
        raise AssertionError("cache option should use the cache-aware RT entry point")

    def cached(config, *, cache_directory, seed):
        captured.update({"config": config, "cache_directory": cache_directory, "seed": seed})
        return FakeDataset()

    monkeypatch.setattr(sionna2, "generate_raytraced", unexpected_uncached)
    monkeypatch.setattr(sionna2, "generate_raytraced_cached", cached)
    monkeypatch.setattr(
        "sys.argv",
        [
            "loslab",
            "rt-data",
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


def test_experiment_config_rejects_unknown_keys(tmp_path, monkeypatch) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "schema: loslab.experiment.v1\ninputs: [data.npz]\noutput: run\ntyop: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["loslab", "experiment", "--config", str(config)])

    with pytest.raises(ValueError, match="Unknown experiment config keys"):
        main()


def test_experiment_config_dispatches_resolved_options(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "schema: loslab.experiment.v1\ninputs: [data.npz]\noutput: run\n"
        "methods: [logreg]\nseeds: [13]\nbootstrap: 10\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(inputs, output, **options):
        calls.append((inputs, output, options))
        return {"status": "ok"}

    monkeypatch.setattr("loslab.experiment.run_experiment", fake_run)
    monkeypatch.setattr("sys.argv", ["loslab", "experiment", "--config", str(config)])

    main()

    inputs, output, options = calls[0]
    assert inputs == ["data.npz"]
    assert output == "run"
    assert options["methods"] == ["logreg"]
    assert options["seeds"] == [13]
    assert options["bootstrap"] == 10
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}

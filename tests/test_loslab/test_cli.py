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

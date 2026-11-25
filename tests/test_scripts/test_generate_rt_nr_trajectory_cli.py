from __future__ import annotations

import json

import pytest
from scripts import generate_rt_nr_trajectory as trajectory_cli


def _write_spec(path, **overrides):
    values = {
        "trajectory_id": "fixture-trajectory",
        "scene": "simple_reflector",
        "transmitter_position_m": [0, 0, 3],
        "receiver_start_m": [1, 0, 1],
        "receiver_velocity_mps": [0, 0, 0],
    }
    values.update(overrides)
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _argv(spec, store, *options):
    return ["generate_rt_nr_trajectory", str(spec), str(store), *options]


def test_trajectory_cli_reports_missing_spec(tmp_path, monkeypatch, capsys) -> None:
    spec = tmp_path / "missing.json"
    monkeypatch.setattr("sys.argv", _argv(spec, tmp_path / "store"))

    with pytest.raises(SystemExit) as error:
        trajectory_cli.main()

    assert error.value.code == 2
    assert "trajectory spec does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "trajectory spec root must be a JSON object"),
        (
            {
                "trajectory_id": "fixture",
                "scene": "simple_reflector",
                "transmitter_position_m": [0, 0],
                "receiver_start_m": [1, 0, 1],
                "receiver_velocity_mps": [0, 0, 0],
            },
            "transmitter_position_m must be a JSON array with three numbers",
        ),
    ],
)
def test_trajectory_cli_reports_invalid_spec_payloads(tmp_path, monkeypatch, capsys, payload, message):
    spec = tmp_path / "invalid.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("sys.argv", _argv(spec, tmp_path / "store", "--pilot"))

    with pytest.raises(SystemExit) as error:
        trajectory_cli.main()

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_trajectory_cli_enforces_pilot_step_limit(tmp_path, monkeypatch, capsys) -> None:
    spec = _write_spec(tmp_path / "long.json", num_steps=101)
    monkeypatch.setattr("sys.argv", _argv(spec, tmp_path / "store", "--pilot"))

    with pytest.raises(SystemExit) as error:
        trajectory_cli.main()

    assert error.value.code == 2
    assert "pilot trajectory generation is bounded" in capsys.readouterr().err


def test_trajectory_cli_reports_generator_errors_without_native_work(
    tmp_path, monkeypatch, capsys
) -> None:
    spec = _write_spec(tmp_path / "request.json")
    monkeypatch.setattr(
        trajectory_cli,
        "generate_rt_nr_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture generator refusal")),
    )
    monkeypatch.setattr("sys.argv", _argv(spec, tmp_path / "store", "--pilot"))

    with pytest.raises(SystemExit) as error:
        trajectory_cli.main()

    assert error.value.code == 2
    assert "fixture generator refusal" in capsys.readouterr().err

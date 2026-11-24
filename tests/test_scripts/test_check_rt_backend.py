"""RT backend probe command: reports the selected Mitsuba variant or refuses cleanly."""

from __future__ import annotations

import json

import pytest

import scripts.check_rt_backend as check_rt_backend


class _Selection:
    def as_dict(self) -> dict:
        return {"backend": "llvm", "variant": "llvm_ad_mono_polarized"}


def test_successful_probe_prints_the_selected_variant(monkeypatch, capsys) -> None:
    activated = []
    monkeypatch.setattr(check_rt_backend, "select_rt_backend", lambda backend: _Selection())
    monkeypatch.setattr(check_rt_backend, "activate_rt_backend", lambda selection: activated.append(selection))
    monkeypatch.setattr("sys.argv", ["check_rt_backend", "--backend", "llvm"])
    check_rt_backend.main()
    assert json.loads(capsys.readouterr().out) == {"backend": "llvm", "variant": "llvm_ad_mono_polarized"}
    assert len(activated) == 1


def test_unavailable_backend_is_reported_through_the_parser(monkeypatch, capsys) -> None:
    def refuse(backend):
        raise RuntimeError("no CUDA variant is installed")

    monkeypatch.setattr(check_rt_backend, "select_rt_backend", refuse)
    monkeypatch.setattr("sys.argv", ["check_rt_backend", "--backend", "cuda"])
    with pytest.raises(SystemExit) as error:
        check_rt_backend.main()
    assert error.value.code == 2
    assert "no CUDA variant is installed" in capsys.readouterr().err

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import scripts.build_standardized_pusch_dataset as command


@dataclass(frozen=True)
class FakeManifest:
    sample_id: str
    sha256: str = "a" * 64


@pytest.mark.parametrize("family", ["tdl", "cdl"])
def test_main_builds_receiver_records_and_immutable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    family: str,
) -> None:
    output = tmp_path / "records"
    channel = type("FakeChannel", (), {"metadata": {"profile_proxy": "A"}})()
    run = type("FakeRun", (), {"run_id": "fixture-run"})()
    records = [object(), object()]
    generated: dict[str, object] = {}
    if family == "tdl":
        import integrations.sionna.tdl as adapter

        monkeypatch.setattr(adapter, "generate_tdl_frequency_response", lambda **kwargs: channel)
    else:
        import integrations.sionna.cdl as adapter

        monkeypatch.setattr(adapter, "generate_cdl_frequency_response", lambda **kwargs: channel)
    monkeypatch.setattr(
        command,
        "standardized_run_from_channel",
        lambda actual_channel, *, family: (
            generated.update(channel=actual_channel, family=family) or run
        ),
    )
    monkeypatch.setattr(
        command, "build_standardized_pusch_dmrs_records", lambda *args, **kwargs: records
    )
    written: list[object] = []

    def write_record(root: Path, record: object) -> FakeManifest:
        written.append((root, record))
        return FakeManifest(f"sample-{len(written)}")

    monkeypatch.setattr(command, "write_nr_record", write_record)
    monkeypatch.setattr(command, "versions", lambda: {"git_sha": "fixture-sha", "python": "3.12"})
    monkeypatch.setattr(command, "package_source_sha256", lambda: "b" * 64)
    monkeypatch.setattr(command, "sha256_file", lambda _path: "c" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build", str(output), "--family", family, "--samples", "2", "--seed", "41"],
    )

    command.main()

    report = json.loads((output / "build-report.json").read_text())
    assert report["family"] == family
    assert report["run_id"] == "fixture-run"
    assert report["samples"] == 2
    assert report["profile_label_semantics"] == "proxy only; not geometric LOS/NLOS"
    assert report["provenance"]["source_revision"] == "fixture-sha"
    assert report["record_manifests"][0]["manifest_path"].startswith("manifests/")
    assert generated == {"channel": channel, "family": command.StandardizedFamily(family)}
    assert len(written) == 2
    assert json.loads(capsys.readouterr().out)["run_id"] == "fixture-run"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--samples", "0"], "--samples must be positive"),
        (["--seed", "-1"], r"--seed must be in \[0, 2\*\*63 - 3\]"),
        (["--seed", str(2**63 - 2)], r"--seed must be in \[0, 2\*\*63 - 3\]"),
    ],
)
def test_main_rejects_invalid_sample_count_and_seed(
    monkeypatch: pytest.MonkeyPatch, options: list[str], message: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["build", "out", *options])
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2


def test_publish_immutable_json_accepts_exact_replay_and_refuses_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "report.json"
    payload = {"status": "PASS", "count": 2}

    command._publish_immutable_json(path, payload)
    command._publish_immutable_json(path, payload)

    assert json.loads(path.read_text()) == payload
    with pytest.raises(ValueError, match="refusing to replace an immutable build report"):
        command._publish_immutable_json(path, {"status": "DIFFERENT"})

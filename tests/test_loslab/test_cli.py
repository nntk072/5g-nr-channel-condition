from __future__ import annotations

import json

import numpy as np
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


def test_g3_path_roster_audit_dispatches_to_packaged_evaluator(monkeypatch, tmp_path) -> None:
    from evaluation import g3_path_equivalence_roster_cli

    input_path = tmp_path / "roster.json"
    captured: dict[str, object] = {}

    def audit(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(g3_path_equivalence_roster_cli, "main", audit)
    monkeypatch.setattr("sys.argv", ["loslab", "g3-path-roster-audit", str(input_path)])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 0
    assert captured["argv"] == [str(input_path)]


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

    def fake_generate(config, *, seed, **_kwargs):
        captured["config"] = config
        captured["seed"] = seed
        return FakeDataset()

    monkeypatch.setattr(sionna2, "generate_raytraced", fake_generate)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("sys.argv", ["loslab", "rt-data", "--pilot", "--output", "unused.npz"])

    main()

    assert captured["config"].mitsuba_variant == "auto"
    assert captured["seed"] == 42
    assert "GEOMETRY-LABELED RT" in capsys.readouterr().out


def test_rt_data_cli_persists_generation_failure_report_without_writing_dataset(
    tmp_path, monkeypatch, capsys
) -> None:
    import loslab.sionna2 as sionna2

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
    monkeypatch.setattr("sys.argv", ["loslab", "rt-data", "--pilot", "--output", str(output_path)])

    with pytest.raises(sionna2.RaytraceGenerationError, match="No NLOS samples"):
        main()

    assert not output_path.exists()
    assert json.loads(failure_path.read_text(encoding="utf-8")) == report
    assert str(failure_path) in capsys.readouterr().err


def test_rt_data_cli_does_not_overwrite_existing_failure_report(monkeypatch, tmp_path) -> None:
    import loslab.sionna2 as sionna2

    output_path = tmp_path / "failed.npz"
    failure_path = tmp_path / "failed.npz.failure.json"
    failure_path.write_text("preserve", encoding="utf-8")

    def fail_generation(config, *, seed, **_kwargs):
        raise sionna2.RaytraceGenerationError("No NLOS samples", {"status": "failed"})

    monkeypatch.setattr(sionna2, "generate_raytraced", fail_generation)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("sys.argv", ["loslab", "rt-data", "--pilot", "--output", str(output_path)])

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()

    assert failure_path.read_text(encoding="utf-8") == "preserve"


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

    def cached(config, *, cache_directory, seed, **_kwargs):
        captured.update({"config": config, "cache_directory": cache_directory, "seed": seed})
        return FakeDataset()

    monkeypatch.setattr(sionna2, "generate_raytraced", unexpected_uncached)
    monkeypatch.setattr(sionna2, "generate_raytraced_cached", cached)
    monkeypatch.setattr(sionna2, "authorize_rt_data_generation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "sys.argv",
        [
            "loslab",
            "rt-data",
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


def test_nr_observe_cli_builds_an_immutable_dmrs_input_dataset(
    tmp_path, monkeypatch, capsys
) -> None:
    import integrations.sionna.rt_dmrs as rt_dmrs

    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "observed.npz"
    source = ChannelDataset(
        h=np.ones((2, 1, 16), dtype=np.complex64),
        y=np.asarray([0, 1], dtype=np.uint8),
        group=np.asarray(["g0", "g1"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["sionna_rt", "sionna_rt"]),
        metadata=tuple(
            {
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "grid_config": {"n_subcarriers": 16, "spacing_hz": 30_000.0},
            }
            for _ in range(2)
        ),
    )
    source.save(source_path)
    observed = ChannelDataset(
        h=source.h,
        y=source.y,
        group=source.group,
        domain=source.domain,
        source=source.source,
        metadata=source.metadata,
        label_semantics="geometric_direct_path_oracle",
        generation_report={"observation_semantics": "Sionna PUSCH DMRS LS estimate"},
    )
    captured = {}

    def fake_build(datasets, **options):
        captured["datasets"] = datasets
        captured["options"] = options
        return observed

    monkeypatch.setattr(rt_dmrs, "build_rt_dmrs_observation_dataset", fake_build)
    monkeypatch.setattr(
        "sys.argv",
        [
            "loslab",
            "nr-observe",
            "--pilot",
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--snr-db",
            "8",
            "--seed",
            "31",
            "--device",
            "cuda:0",
        ],
    )

    main()

    assert len(captured["datasets"]) == 1
    assert captured["options"]["snr_db"] == 8.0
    assert captured["options"]["seed"] == 31
    assert captured["options"]["device"] == "cuda:0"
    assert captured["options"]["n_size_bwp"] == 6
    assert captured["options"]["subcarrier_spacing_khz"] == 30
    assert captured["options"]["authorization"].mode == "pilot"
    assert len(captured["options"]["source_artifacts"]) == 1
    assert ChannelDataset.load(output_path).label_semantics == "geometric_direct_path_oracle"
    assert json.loads(capsys.readouterr().out)["evidence_grade"] == "SIMULATION_ONLY_UNVALIDATED"


def test_nr_observe_cli_refuses_to_overwrite_existing_dataset(tmp_path, monkeypatch):
    output_path = tmp_path / "existing.npz"
    output_path.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["loslab", "nr-observe", "--input", "missing.npz", "--output", str(output_path)],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()

    assert output_path.read_text(encoding="utf-8") == "preserve"


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
        "methods: [logreg]\nseeds: [13]\nbootstrap: 10\ncapacity: large\n"
        "split_lock_path: lock.json\nsplit_signature_path: lock.json.sig\n"
        "trusted_split_signers_path: allowed_signers\nsplit_signer_id: custodian@example.org\n"
        "prior_shift:\n"
        "  schema: loslab.prior_shift.v1\n"
        "  training_prior_kind: enriched_stress\n"
        "  training_prior_id: enriched-v1\n"
        "  target_prior_kind: natural\n"
        "  target_prior_id: field-v1\n"
        "  target_positive_prevalence: 0.2\n"
        f"  target_prior_evidence_sha256: {'a' * 64}\n",
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
    assert options["capacity"] == "large"
    assert options["split_lock_path"] == "lock.json"
    assert options["split_signature_path"] == "lock.json.sig"
    assert options["trusted_split_signers_path"] == "allowed_signers"
    assert options["split_signer_id"] == "custodian@example.org"
    assert options["prior_shift"] == {
        "schema": "loslab.prior_shift.v1",
        "training_prior_kind": "enriched_stress",
        "training_prior_id": "enriched-v1",
        "target_prior_kind": "natural",
        "target_prior_id": "field-v1",
        "target_positive_prevalence": 0.2,
        "target_prior_evidence_sha256": "a" * 64,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}

from __future__ import annotations

import json

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from scripts.generate_rt_dmrs_observations import main


def test_nr_observe_cli_builds_an_immutable_dmrs_input_dataset(
    tmp_path, monkeypatch, capsys
) -> None:
    import integrations.sionna.rt_dmrs as rt_dmrs

    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "observed.npz"
    source = ChannelDataset(
        h=np.ones((2, 1, 16), dtype=np.complex64),
        y=np.asarray([0, 1], dtype=np.uint8),
        group=np.asarray(["group0", "group1"]),
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
            "generate_rt_dmrs_observations",
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
        ["generate_rt_dmrs_observations", "--input", "missing.npz", "--output", str(output_path)],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()

    assert output_path.read_text(encoding="utf-8") == "preserve"

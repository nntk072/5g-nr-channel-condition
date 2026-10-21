from __future__ import annotations

import pytest
from configuration.loader import load_config, resolve_config


def test_load_yaml_mapping(tmp_path) -> None:
    source = tmp_path / "experiment.yaml"
    source.write_text("seed: 42\nmethod: mlp\n", encoding="utf-8")
    assert load_config(source) == {"seed": 42, "method": "mlp"}


def test_load_config_rejects_non_mapping(tmp_path) -> None:
    source = tmp_path / "invalid.yaml"
    source.write_text("- value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(source)


def test_resolved_config_identity_is_deterministic() -> None:
    first = resolve_config({"seed": 42, "method": "mlp"})
    second = resolve_config({"method": "mlp", "seed": 42})
    assert first["sha256"] == second["sha256"]
    assert first["schema"] == "loslab.config.v1"

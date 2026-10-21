"""Load YAML/JSON configuration and attach a stable identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a mapping from a YAML or JSON configuration file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = (
            yaml.safe_load(stream)
            if source.suffix.lower() in {".yaml", ".yml"}
            else json.load(stream)
        )
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {source}")
    return value


def resolve_config(config: dict[str, Any], *, source: str | Path | None = None) -> dict[str, Any]:
    """Return a normalized config and canonical SHA256 without mutating input."""
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    result: dict[str, Any] = {
        "schema": "loslab.config.v1",
        "values": json.loads(serialized),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    if source is not None:
        result["source"] = str(Path(source))
    return result

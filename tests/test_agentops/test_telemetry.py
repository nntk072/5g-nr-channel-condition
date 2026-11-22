import json
from pathlib import Path

from agentops.telemetry import append_event, normalize_provider_usage


def test_unknown_provider_usage_stays_unknown() -> None:
    assert normalize_provider_usage(None)["prompt_tokens"] == "unknown"


def test_telemetry_redacts_credentials(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_event(path, {"message": "token=secret-value", "status": "failed"})
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "secret-value" not in stored["message"]

"""Each recorded historical source digest must match the blob of its recorded commit."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from artifacts.rebinding import historical_revision

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs" / "reference" / "historical-source-revisions.json"


def test_each_recorded_revision_matches_its_commit_blob() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    assert payload["entries"]
    for item in payload["entries"]:
        blob = subprocess.run(
            ["git", "show", f"{item['commit']}:{item['path']}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == item["sha256"], item


def test_lookup_returns_commit_only_for_recorded_digest() -> None:
    assert (
        historical_revision(
            "receivers/nr/pusch.py",
            "d756599070c350e65d96b823738c98ad7be59d9188eb7eba98683d748716594d",
            ROOT,
        )
        == "3034710"
    )
    assert historical_revision("receivers/nr/pusch.py", "0" * 64, ROOT) is None

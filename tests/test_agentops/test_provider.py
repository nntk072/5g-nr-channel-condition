import sys
from pathlib import Path

import pytest
from agentops.provider import ProviderError, _redact_structured, redact, run_provider


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "provider.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_provider_success_and_secret_redaction(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        "import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'ok': p['task']}))\nprint('token=secret-value Bearer standalone-stderr-secret', file=sys.stderr)\n",
    )
    result = run_provider(sys.executable, [str(script)], {"task": "A"})
    assert result["stdout"] == {"ok": "A"}
    assert "secret-value" not in result["stderr"]
    assert "standalone-stderr-secret" not in result["stderr"]


def test_provider_redacts_structured_json_secrets_and_argv(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        "import json\nprint(json.dumps({'nested': {'token': 'stdout-secret', 'safe': 'ok', 'message': 'Bearer standalone-stdout-bearer'}, 'authorization': 'Bearer stdout-bearer'}))\n",
    )
    result = run_provider(
        sys.executable,
        [str(script), "--api-key", "argv-secret"],
        {},
    )
    rendered = repr(result)
    assert "stdout-secret" not in rendered
    assert "stdout-bearer" not in rendered
    assert "standalone-stdout-bearer" not in rendered
    assert "argv-secret" not in rendered
    assert result["stdout"]["nested"]["safe"] == "ok"


def test_redact_structured_redacts_bearer_in_ordinary_string_values() -> None:
    value = {
        "message": "Bearer structured-secret",
        "nested": ["safe", "token=inner-secret"],
    }
    redacted = _redact_structured(value)
    rendered = repr(redacted)
    assert "structured-secret" not in rendered
    assert "inner-secret" not in rendered
    assert redacted["nested"][0] == "safe"


def test_provider_failure_receipt_redacts_json_and_bearer_secrets(tmp_path: Path) -> None:
    crash = _script(
        tmp_path,
        'import sys\nprint(\'{"token":"stderr-secret","authorization":"Bearer bearer-secret"}\', file=sys.stderr)\nprint("Bearer standalone-failure-secret", file=sys.stderr)\nraise SystemExit(9)\n',
    )
    with pytest.raises(ProviderError) as error:
        run_provider(sys.executable, [str(crash), "--password=argv-password"], {})
    assert error.value.receipt is not None
    rendered = repr(error.value.receipt)
    assert "stderr-secret" not in rendered
    assert "bearer-secret" not in rendered
    assert "standalone-failure-secret" not in rendered
    assert "argv-password" not in rendered


def test_provider_runs_in_explicit_working_directory(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        "import json,os\nprint(json.dumps({'cwd': os.getcwd()}))\n",
    )
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    result = run_provider(sys.executable, [str(script)], {}, cwd=workdir)
    assert result["stdout"] == {"cwd": str(workdir)}


def test_provider_rejects_malformed_and_timeout(tmp_path: Path) -> None:
    bad = _script(tmp_path, "print('not-json')\n")
    with pytest.raises(ProviderError, match="valid JSON"):
        run_provider(sys.executable, [str(bad)], {})
    slow = _script(tmp_path, "import time\ntime.sleep(2)\nprint('{}')\n")
    with pytest.raises(ProviderError, match="timed out") as timeout_error:
        run_provider(sys.executable, [str(slow)], {}, timeout_seconds=1)
    assert timeout_error.value.receipt is not None
    assert timeout_error.value.receipt["error"] == "timeout"


def test_provider_rejects_crash_and_truncated_output(tmp_path: Path) -> None:
    crash = _script(tmp_path, "import sys\nprint('boom', file=sys.stderr)\nraise SystemExit(9)\n")
    with pytest.raises(ProviderError, match="code 9") as crash_error:
        run_provider(sys.executable, [str(crash)], {})
    assert crash_error.value.receipt is not None
    assert crash_error.value.receipt["exit_code"] == 9

    large = _script(
        tmp_path,
        "import json\nprint(json.dumps({'payload': 'x' * 4096}))\n",
    )
    with pytest.raises(ProviderError, match="stdout exceeded"):
        run_provider(sys.executable, [str(large)], {}, output_limit_bytes=1024)


def test_provider_permission_error_is_contained(monkeypatch) -> None:
    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("agentops.provider.subprocess.run", denied)
    with pytest.raises(ProviderError, match="execution failed: PermissionError"):
        run_provider(sys.executable, [], {})


def test_redact_handles_common_secret_labels() -> None:
    text = redact(
        'api_key=abc Authorization: Bearer bearer123 password=hunter2 "token":"json-secret" '
        "upstream=failed Bearer standalone-bearer"
    )
    assert "abc" not in text
    assert "bearer123" not in text
    assert "hunter2" not in text
    assert "json-secret" not in text
    assert "standalone-bearer" not in text


def test_provider_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = _script(tmp_path, 'print(\'{"a": 1, "a": 2}\')\n')
    with pytest.raises(ProviderError, match="strict valid JSON"):
        run_provider(sys.executable, [str(duplicate)], {})
    nonfinite = _script(tmp_path, "print('{\"a\": NaN}')\n")
    with pytest.raises(ProviderError, match="strict valid JSON"):
        run_provider(sys.executable, [str(nonfinite)], {})


def test_provider_unavailable_executable_is_reported_distinctly(tmp_path: Path) -> None:
    missing = tmp_path / "missing-provider"
    with pytest.raises(ProviderError, match="provider executable is unavailable"):
        run_provider(str(missing), [], {})

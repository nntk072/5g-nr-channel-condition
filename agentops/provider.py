"""Bounded provider invocation with strict argv execution and redacted receipts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, canonical_json_bytes, load_strict_json_bytes

_SAFE_ENV = ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
_SECRET_LABELS = r"secret[_-]?access[_-]?key|session[_-]?token|auth(?:orization)?|x[_-]?api[_-]?key|api[_-]?key|token|secret|password|passwd|credentials?|cookies?|set[_-]cookie|private[_-]key|secret[_-]key|aws_secret_access_key|access[_-]?key|oauth2[_-]?bearer"
_SECRET = re.compile(
    r"(?i)([\"']?(?:" + _SECRET_LABELS + r")[\"']?"
    r"\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|Digest\s+[^\r\n]+|(?:Bearer|Basic|Token|ApiKey)\s+[^\s,;}\]]+"
    r"|[^\s,;}\]]+)"
)
_SECRET_KEY = re.compile(r"(?i)^(?:" + _SECRET_LABELS + r")$")
_SECRET_KEY_SUFFIXES = (
    "_token",
    "_secret",
    "_password",
    "_api_key",
    "_cookie",
    "_private_key",
    "_secret_key",
)
_URL_USERINFO = re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;}\]\x22\x27]+")
# Standalone credentials with a well-known prefix, even without a label.
_KNOWN_CREDENTIAL = re.compile(
    r"\b(?:sk-(?:ant-)?[A-Za-z0-9_\-]{8,512}|gh[pousr]_[A-Za-z0-9]{20,255}|"
    r"github_pat_[A-Za-z0-9_]{20,255}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,255}|"
    r"AIza[0-9A-Za-z_-]{20,128}|eyJ[A-Za-z0-9_-]{4,512}\.[A-Za-z0-9_-]{4,512}\.[A-Za-z0-9_-]{1,1024})\b"
)


class ProviderError(ContractError):
    """Raised when a configured provider cannot produce a bounded valid response."""

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = receipt


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_COOKIE_HEADER = re.compile(r"(?i)((?<![A-Za-z0-9\"'])(?:set-)?cookie\s*[:=]\s*)[^\r\n]+")


def redact(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(2)
        if value.startswith("[REDACTED"):
            # Already redacted by an earlier pass; leave the text exactly as it is.
            return match.group(0)
        if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
            replacement = f"{value[0]}[REDACTED]{value[0]}"
        elif re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|null|true|false", value):
            # Bare JSON scalars are replaced with a quoted placeholder so that the redacted
            # JSON still parses and the response is not rejected as malformed.
            replacement = '"[REDACTED]"'
        else:
            replacement = "[REDACTED]"
        return f"{match.group(1)}{replacement}"

    # Cookie values run to the end of the header line, so every cookie pair is covered.
    text = _COOKIE_HEADER.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    redacted = _BEARER.sub("Bearer [REDACTED]", _SECRET.sub(replace, text))
    redacted = _URL_USERINFO.sub("[REDACTED]", redacted)
    return _KNOWN_CREDENTIAL.sub("[REDACTED]", redacted)


def _is_secret_key(key: str) -> bool:
    # Split camelCase (accessToken -> access_Token) before lowering so that camelCase and
    # snake_case forms of the same credential name are treated alike.
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = split.lower().replace("-", "_")
    return (
        bool(_SECRET_KEY.fullmatch(normalized))
        or normalized.endswith(_SECRET_KEY_SUFFIXES)
        # All-lowercase concatenations such as accesstoken or clientsecret.
        or normalized.endswith(("token", "secret", "password", "apikey", "passwd", "cookie"))
    )


def _redact_structured(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            redact(str(key)): "[REDACTED]" if _is_secret_key(str(key)) else _redact_structured(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_structured(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    return value


def _argv_summary(argv: list[str]) -> list[str]:
    """Receipt form of a command: the executable name and the argument count.

    Argument values are not persisted. Provider argv is static configuration, and recording it
    per argument invited leaks that were hard to bound. The provider configuration is the audit
    source for the arguments.
    """
    if not argv:
        return []
    return [Path(argv[0]).name, f"<{len(argv) - 1} arguments>"]


def _resolve_executable(executable: str) -> str:
    candidate = Path(executable)
    if candidate.is_absolute():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ProviderError(f"provider executable is unavailable: {executable}")
        return str(candidate)
    resolved = shutil.which(executable)
    if resolved is None:
        raise ProviderError(f"provider executable is unavailable: {executable}")
    return resolved


def _bounded(text: str, limit: int) -> tuple[str, bool]:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text, False
    return raw[:limit].decode("utf-8", errors="replace"), True


def _output_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _omitted(text: str) -> str:
    """Describe provider output without persisting its content.

    Pattern redaction cannot be made exhaustive for free text, so receipts record only the
    size and digest of raw provider output. Parsed success responses are still persisted,
    after structured redaction.
    """
    raw = text.encode("utf-8", errors="replace")
    return f"[OMITTED: {len(raw)} bytes, sha256 {hashlib.sha256(raw).hexdigest()}]"


_MAX_JSON_DEPTH = 256


def _json_depth(value: Any) -> int:
    """Nesting depth of a parsed JSON value, computed without recursion."""
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            deepest = max(deepest, depth)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            deepest = max(deepest, depth)
            stack.extend((child, depth + 1) for child in item)
    return deepest


def _redact_structured_or_fail(
    response: Any,
    resolved: str,
    argv: list[str],
    command_digest: str,
    started: str,
    ended: str,
    exit_code: int | None,
    stdout_text: str,
    stderr_text: str,
    output_limit_bytes: int,
) -> Any:
    try:
        return _redact_structured(response)
    except RecursionError as error:
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=exit_code,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            output_limit_bytes=output_limit_bytes,
            error="nesting_limit",
        )
        raise ProviderError("provider response nests too deeply", receipt=receipt) from error


def _failure_receipt(
    *,
    resolved: str,
    argv: list[str],
    command_digest: str,
    started: str,
    ended: str,
    exit_code: int | None,
    stdout_text: str,
    stderr_text: str,
    output_limit_bytes: int,
    error: str,
) -> dict[str, Any]:
    return {
        "provider": Path(resolved).name,
        "argv": _argv_summary(argv),
        "command_sha256": command_digest,
        "started_at": started,
        "ended_at": ended,
        "exit_code": exit_code,
        "stdout_raw": _omitted(stdout_text),
        "stderr": _omitted(stderr_text),
        "stdout_truncated": len(stdout_text.encode("utf-8", errors="replace")) > output_limit_bytes,
        "stderr_truncated": len(stderr_text.encode("utf-8", errors="replace")) > output_limit_bytes,
        "error": error,
    }


def run_provider(
    executable: str,
    args: list[str] | tuple[str, ...],
    payload: Mapping[str, Any],
    *,
    env_allowlist: tuple[str, ...] = (),
    timeout_seconds: int = 300,
    output_limit_bytes: int = 1_000_000,
    inherited_env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    resolved = _resolve_executable(executable)
    if not 1 <= timeout_seconds <= 3600:
        raise ProviderError("provider timeout is outside the allowed range")
    if not 1024 <= output_limit_bytes <= 16_777_216:
        raise ProviderError("provider output limit is outside the allowed range")
    source_env = dict(os.environ if inherited_env is None else inherited_env)
    names = set(_SAFE_ENV) | set(env_allowlist)
    env = {name: source_env[name] for name in names if name in source_env}
    argv = [resolved, *args]
    command_digest = hashlib.sha256(canonical_json_bytes(_argv_summary(argv))).hexdigest()
    input_bytes = canonical_json_bytes(dict(payload)) + b"\n"
    started = _now()
    try:
        result = subprocess.run(
            argv,
            input=input_bytes,
            cwd=cwd,
            env=env,
            shell=False,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        ended = _now()
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=None,
            stdout_text=_output_text(error.stdout),
            stderr_text=_output_text(error.stderr),
            output_limit_bytes=output_limit_bytes,
            error="timeout",
        )
        raise ProviderError(
            f"provider timed out after {timeout_seconds}s", receipt=receipt
        ) from error
    except OSError as error:
        ended = _now()
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=None,
            stdout_text="",
            stderr_text=str(error),
            output_limit_bytes=output_limit_bytes,
            error=type(error).__name__,
        )
        raise ProviderError(
            f"provider execution failed: {type(error).__name__}", receipt=receipt
        ) from error
    ended = _now()
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    stdout_truncated = len(result.stdout) > output_limit_bytes
    stderr_truncated = len(result.stderr) > output_limit_bytes
    if result.returncode != 0:
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=result.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            output_limit_bytes=output_limit_bytes,
            error="nonzero_exit",
        )
        raise ProviderError(
            f"provider exited with code {result.returncode}; stderr is retained only as a digest",
            receipt=receipt,
        )
    if stdout_truncated or stderr_truncated:
        stream = "stdout" if stdout_truncated else "stderr"
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=result.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            output_limit_bytes=output_limit_bytes,
            error=f"{stream}_limit",
        )
        raise ProviderError(
            f"provider {stream} exceeded the configured output limit", receipt=receipt
        )
    try:
        # Parse the bounded raw text, then redact the parsed structure. Redacting raw text
        # before parsing corrupts JSON escaping and leaves nested credential values exposed.
        raw_stdout, _ = _bounded(stdout_text, output_limit_bytes)
        response = load_strict_json_bytes(raw_stdout.encode("utf-8"))
        if _json_depth(response) > _MAX_JSON_DEPTH:
            raise ContractError("provider response nests too deeply")
    except (ContractError, RecursionError) as error:
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=result.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            output_limit_bytes=output_limit_bytes,
            error="malformed_json",
        )
        raise ProviderError("provider stdout is not strict valid JSON", receipt=receipt) from error
    if not isinstance(response, dict):
        receipt = _failure_receipt(
            resolved=resolved,
            argv=argv,
            command_digest=command_digest,
            started=started,
            ended=ended,
            exit_code=result.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            output_limit_bytes=output_limit_bytes,
            error="non_object_json",
        )
        raise ProviderError("provider response must be a JSON object", receipt=receipt)
    return {
        "provider": Path(resolved).name,
        "argv": _argv_summary(argv),
        "command_sha256": command_digest,
        "started_at": started,
        "ended_at": ended,
        "exit_code": result.returncode,
        "stdout": _redact_structured_or_fail(
            response,
            resolved,
            argv,
            command_digest,
            started,
            ended,
            result.returncode,
            stdout_text,
            stderr_text,
            output_limit_bytes,
        ),
        "stderr": _omitted(stderr_text),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }

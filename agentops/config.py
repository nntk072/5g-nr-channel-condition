"""Strict configuration loading for the engineering control plane."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import ROOT, ContractError, reject_nonfinite, validate_relative_path


@dataclass(frozen=True)
class ProviderConfig:
    executable: str
    args: tuple[str, ...]
    env_allowlist: tuple[str, ...]
    timeout_seconds: int
    output_limit_bytes: int


@dataclass(frozen=True)
class AgentConfig:
    task_cards: tuple[str, ...]
    state_root: str
    log_root: str
    provider: ProviderConfig | None
    reviewer_provider: ProviderConfig | None = None


def _clean_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ContractError(f"{label} must be a nonempty trimmed string")
    return value


def _resolved_executable(executable: str) -> str:
    """Return the real path of the executable a provider configuration would launch."""
    candidate = Path(executable)
    if candidate.is_absolute():
        return os.path.realpath(candidate)
    found = shutil.which(executable)
    return os.path.realpath(found) if found else executable


def _parse_provider(provider_value: Any, label: str) -> ProviderConfig | None:
    if provider_value is None:
        return None
    if not isinstance(provider_value, dict):
        raise ContractError(f"{label} configuration must be a mapping or null")
    required = {"executable", "args", "env_allowlist", "timeout_seconds", "output_limit_bytes"}
    if set(provider_value) != required:
        raise ContractError(f"{label} configuration keys do not match the canonical contract")
    executable = _clean_string(provider_value["executable"], f"{label} executable")
    args = provider_value["args"]
    env_allowlist = provider_value["env_allowlist"]
    timeout_seconds = provider_value["timeout_seconds"]
    output_limit_bytes = provider_value["output_limit_bytes"]
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ContractError(f"{label} args must be a string list")
    if not isinstance(env_allowlist, list) or not all(
        isinstance(item, str) and item for item in env_allowlist
    ):
        raise ContractError(f"{label} env_allowlist must be a string list")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
        raise ContractError(f"{label} timeout_seconds is outside the allowed range")
    if not isinstance(output_limit_bytes, int) or not 1024 <= output_limit_bytes <= 16_777_216:
        raise ContractError(f"{label} output_limit_bytes is outside the allowed range")
    return ProviderConfig(
        executable=executable,
        args=tuple(args),
        env_allowlist=tuple(env_allowlist),
        timeout_seconds=timeout_seconds,
        output_limit_bytes=output_limit_bytes,
    )


def load_config(
    path: str | Path = "config/agentops/default.yaml", repo_root: Path = ROOT
) -> AgentConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ContractError("cannot load agentops configuration") from error
    reject_nonfinite(value)
    if not isinstance(value, dict):
        raise ContractError("agentops configuration must be a mapping")
    allowed = {"schema", "task_cards", "state_root", "log_root", "provider", "reviewer_provider"}
    if set(value) - allowed:
        raise ContractError(f"unknown agentops configuration keys: {sorted(set(value) - allowed)}")
    if value.get("schema") != "nr-agentops.config.v1":
        raise ContractError("unsupported agentops configuration schema")
    cards = value.get("task_cards")
    if not isinstance(cards, list) or not cards or not all(isinstance(item, str) for item in cards):
        raise ContractError("task_cards must be a nonempty list of paths")
    checked_cards = tuple(validate_relative_path(item, repo_root) for item in cards)
    state_root = validate_relative_path(
        value.get("state_root", "outputs/agentops/tasks"), repo_root
    )
    log_root = validate_relative_path(value.get("log_root", "logs/agents"), repo_root)
    provider = _parse_provider(value.get("provider"), "provider")
    reviewer_provider = _parse_provider(value.get("reviewer_provider"), "reviewer_provider")
    if provider is not None and reviewer_provider is not None:
        # Identity is a configured check, not a cryptographic one: two distinct binaries can
        # still wrap the same model. What is enforced is that the reviewer never resolves to
        # the implementer's executable, whatever flags or path spelling it uses.
        if _resolved_executable(provider.executable) == _resolved_executable(
            reviewer_provider.executable
        ):
            raise ContractError(
                "reviewer_provider resolves to the same executable as provider; "
                "an implementer cannot act as its own independent reviewer"
            )
    return AgentConfig(
        task_cards=checked_cards,
        state_root=state_root,
        log_root=log_root,
        provider=provider,
        reviewer_provider=reviewer_provider,
    )

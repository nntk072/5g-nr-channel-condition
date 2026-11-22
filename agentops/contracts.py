"""Strict task-card and JSON contract loading."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"


class ContractError(ValueError):
    """Raised when an agent contract fails closed."""


@dataclass(frozen=True)
class LoadedTaskCard:
    data: dict[str, Any]
    path: Path
    raw_sha256: str
    source_revision: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json_bytes(raw: bytes) -> Any:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
        reject_nonfinite(value)
    except ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ContractError("invalid JSON") from error
    return value


def reject_nonfinite(value: Any) -> None:
    """Refuse non-finite numbers in keys or values, iteratively and visiting each container once.

    YAML aliases share nodes, so a naive recursive walk can be exponential on a small input.
    """
    seen: set[int] = set()
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ContractError("non-finite numeric value")
        if isinstance(item, dict):
            if id(item) in seen:
                continue
            seen.add(id(item))
            for key, child in item.items():
                stack.append(key)
                stack.append(child)
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            stack.extend(item)


_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}")


def is_ancestor_revision(candidate: str, head: str, repo_root: Path = ROOT) -> bool:
    """True when ``candidate`` is ``head`` or one of its ancestors.

    Task cards bind to the revision they were authored against. Committing the work they
    describe necessarily moves HEAD, so the card's revision must be HEAD or an ancestor of it,
    rather than equal to HEAD. Anything else, including an unknown object, is not an ancestor.
    """
    if not _GIT_OBJECT_ID.fullmatch(candidate) or not _GIT_OBJECT_ID.fullmatch(head):
        return False
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", candidate, head],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot resolve revision ancestry") from error
    return completed.returncode == 0


def current_git_revision(repo_root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot resolve repository Git revision") from error


def _update_path_state_hash(digest: Any, path: Path) -> None:
    """Hash one filesystem entry without following symlinks.

    Git can report an untracked nested repository as a single directory entry.
    Recursing here keeps worktree/reviewer bindings sensitive to mutations
    anywhere inside such a directory instead of collapsing it to a constant.
    """
    try:
        if path.is_symlink():
            digest.update(b"SYMLINK\0")
            digest.update(os.fsencode(os.readlink(path)))
            return
        if path.is_file():
            digest.update(f"FILE:{path.stat().st_mode & 0o777:o}\0".encode("ascii"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return
        if path.is_dir():
            digest.update(f"DIR:{path.stat().st_mode & 0o777:o}\0".encode("ascii"))
            entries = sorted(path.iterdir(), key=lambda item: os.fsencode(item.name))
            for child in entries:
                digest.update(b"ENTRY\0")
                digest.update(os.fsencode(child.name))
                digest.update(b"\0")
                _update_path_state_hash(digest, child)
                digest.update(b"\0")
            return
        if not path.exists():
            digest.update(b"MISSING\0")
            return
    except OSError as error:
        raise ContractError(f"cannot fingerprint worktree path {path}") from error
    raise ContractError(f"unsupported filesystem entry in worktree: {path}")


def path_state_fingerprint(path: Path) -> str:
    """Return a recursive fingerprint for one worktree path."""
    digest = hashlib.sha256()
    _update_path_state_hash(digest, path)
    return digest.hexdigest()


def current_worktree_digest(repo_root: Path = ROOT) -> str:
    """Hash the complete tracked/untracked worktree delta without reading ignored outputs."""
    commands = (
        ["git", "diff", "--name-only", "-z", "--no-renames", "HEAD"],
        ["git", "ls-files", "-o", "--exclude-standard", "-z"],
    )
    paths: set[str] = set()
    try:
        for argv in commands:
            raw = subprocess.check_output(
                argv,
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            paths.update(item for item in raw.decode("utf-8", errors="strict").split("\0") if item)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot fingerprint repository worktree state") from error

    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        _update_path_state_hash(digest, path)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_relative_path(value: str, repo_root: Path = ROOT) -> str:
    path = Path(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or value in {".", "./"}
    ):
        raise ContractError(f"unsafe repository-relative path: {value!r}")
    candidate = repo_root.joinpath(path)
    parent = candidate
    while parent != repo_root and not parent.exists():
        parent = parent.parent
    if parent.exists() and parent.is_symlink():
        raise ContractError(f"task path crosses a symlink: {value!r}")
    try:
        parent.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise ContractError(f"task path escapes repository: {value!r}") from error
    return path.as_posix()


def load_schema(name: str) -> dict[str, Any]:
    raw = (SCHEMA_DIR / name).read_bytes()
    schema = load_strict_json_bytes(raw)
    if not isinstance(schema, dict):
        raise ContractError(f"schema {name} must be an object")
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def _load_yaml(raw: bytes) -> Any:
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
        reject_nonfinite(value)
    except (UnicodeDecodeError, yaml.YAMLError, RecursionError, ValueError) as error:
        raise ContractError("invalid YAML task card") from error
    return value


def load_task_card(path: str | Path, repo_root: Path = ROOT) -> LoadedTaskCard:
    card_path = Path(path)
    if not card_path.is_absolute():
        card_path = repo_root / card_path
    try:
        resolved = card_path.resolve(strict=True)
        resolved.relative_to(repo_root.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise ContractError("task card must be an existing file inside the repository") from error
    if resolved.is_symlink() or card_path.is_symlink():
        raise ContractError("task card may not be a symlink")
    raw = resolved.read_bytes()
    value = _load_yaml(raw)
    if not isinstance(value, dict):
        raise ContractError("task card must be a mapping")
    schema = load_schema("task-card.v1.schema.json")
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ContractError(
            "task card violates schema: " + "; ".join(error.message for error in errors[:8])
        )
    for field in ("owned_paths", "readonly_paths"):
        value[field] = [validate_relative_path(item, repo_root) for item in value.get(field, [])]
    revision = current_git_revision(repo_root)
    if not is_ancestor_revision(value["source_revision"], revision, repo_root):
        raise ContractError(
            f"stale task card source revision {value['source_revision']} "
            f"is not HEAD or an ancestor of {revision}"
        )
    return LoadedTaskCard(
        data=value,
        path=resolved,
        raw_sha256=sha256_bytes(raw),
        source_revision=revision,
    )

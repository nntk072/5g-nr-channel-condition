"""Immutable, content-addressed build nodes for reproducible research workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NODE_SCHEMA = "loslab.build_node_manifest.v1"
_NODE_SCHEMA_PATH = Path(__file__).resolve().parent / "schema/build-node-manifest.v1.schema.json"


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("build-node identity inputs must be finite canonical JSON") from error


def _relative_output(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError("build-node outputs must be safe relative POSIX paths")
    return path


def _file_digest(path: Path, *, require_single_link: bool = False) -> tuple[str, int]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink file opening")
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or (require_single_link and before_path.st_nlink != 1):
        raise ValueError("build artifact must be a regular file without forbidden links")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (require_single_link and before.st_nlink != 1):
            raise ValueError("build artifact changed type while opening")
        if (before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise ValueError("build artifact changed while opening")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise ValueError("build artifact changed while hashing")
    return digest.hexdigest(), before.st_size


def _stable_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError("build-node manifest must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
            or (before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino)
        ):
            raise ValueError("build-node manifest violates its bounded file contract")
        chunks = bytearray()
        while len(chunks) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
    finally:
        os.close(descriptor)
    if (
        len(chunks) > maximum_bytes
        or len(chunks) != before.st_size
        or before.st_size != after.st_size
        or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
        or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
    ):
        raise ValueError("build-node manifest changed during bounded read")
    return bytes(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class BuildNodeSpec:
    """All identities that determine one immutable workflow node."""

    node_id: str
    node_version: str
    configuration: Mapping[str, Any]
    configuration_schema: Mapping[str, Any]
    source_revision: str
    source_bundle_sha256: str
    input_files: Mapping[str, str]
    dependencies: Mapping[str, str]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("node_id", "node_version", "source_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a canonical non-empty string")
        if not isinstance(self.source_bundle_sha256, str) or not _SHA256.fullmatch(
            self.source_bundle_sha256
        ):
            raise ValueError("source_bundle_sha256 must be lowercase SHA-256 hex")
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("build-node outputs must be non-empty and unique")
        for output in self.outputs:
            _relative_output(output)
        if tuple(sorted(self.outputs)) != self.outputs:
            raise ValueError("build-node outputs must use sorted canonical order")
        for name, digest in (*self.input_files.items(), *self.dependencies.items()):
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("build-node input and dependency names must be canonical")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError(f"build-node digest for {name!r} must be lowercase SHA-256")
        _canonical_json(self.configuration)
        _canonical_json(self.configuration_schema)
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(dict(self.configuration_schema))
        jsonschema.validate(dict(self.configuration), dict(self.configuration_schema))

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": _NODE_SCHEMA,
            "node_id": self.node_id,
            "node_version": self.node_version,
            "configuration": dict(self.configuration),
            "configuration_schema_sha256": hashlib.sha256(
                _canonical_json(self.configuration_schema)
            ).hexdigest(),
            "source_revision": self.source_revision,
            "source_bundle_sha256": self.source_bundle_sha256,
            "input_files": dict(sorted(self.input_files.items())),
            "dependencies": dict(sorted(self.dependencies.items())),
            "outputs": list(self.outputs),
        }

    @property
    def node_key(self) -> str:
        return hashlib.sha256(_canonical_json(self.identity_payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class BuildNodeResult:
    node_key: str
    manifest_path: Path
    artifacts: tuple[dict[str, Any], ...]
    reused: bool


def _verify_input_files(spec: BuildNodeSpec) -> None:
    for raw_path, expected in spec.input_files.items():
        path = Path(raw_path)
        actual, _ = _file_digest(path)
        if actual != expected:
            raise ValueError(f"build-node input changed after its digest was frozen: {path}")


def _read_node_manifest(
    root: Path, node_key: str, *, expected_identity: dict[str, Any] | None = None
) -> BuildNodeResult:
    if not _SHA256.fullmatch(node_key):
        raise ValueError("dependency node key must be lowercase SHA-256")
    manifest_path = root / "nodes" / f"{node_key}.json"
    try:
        manifest_bytes = _stable_bytes(manifest_path, maximum_bytes=16 * 1024 * 1024)
    except FileNotFoundError:
        raise
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cached build-node manifest is not valid JSON") from error
    import jsonschema

    schema = json.loads(_NODE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(manifest, schema)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "node_key", "identity", "artifacts"}
        or manifest.get("schema") != _NODE_SCHEMA
        or manifest.get("node_key") != node_key
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ValueError("build-node manifest does not match its content address")
    identity = manifest["identity"]
    if (
        not isinstance(identity, dict)
        or hashlib.sha256(_canonical_json(identity)).hexdigest() != node_key
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise ValueError("build-node manifest identity differs from its content address")
    artifacts = manifest["artifacts"]
    names = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "size_bytes"}:
            raise ValueError("cached build-node artifact entry is malformed")
        if not isinstance(artifact["path"], str):
            raise ValueError("cached build-node artifact path is malformed")
        relative = _relative_output(artifact["path"])
        names.append(relative.as_posix())
        digest = artifact["sha256"]
        size = artifact["size_bytes"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("cached build-node artifact digest is malformed")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("cached build-node artifact size is malformed")
        object_path = root / "objects" / digest[:2] / digest
        actual, actual_size = _file_digest(object_path, require_single_link=True)
        if actual != digest or actual_size != size:
            raise ValueError(f"cached build-node object failed integrity verification: {digest}")
    if tuple(sorted(names)) != tuple(identity["outputs"]) or len(set(names)) != len(names):
        raise ValueError("build-node output roster differs from its frozen identity")
    return BuildNodeResult(node_key, manifest_path, tuple(artifacts), True)


def _load_existing_node(root: Path, spec: BuildNodeSpec) -> BuildNodeResult | None:
    try:
        return _read_node_manifest(root, spec.node_key, expected_identity=spec.identity_payload)
    except FileNotFoundError:
        return None


def _publish_object(root: Path, source: Path) -> tuple[str, int]:
    digest, size = _file_digest(source, require_single_link=True)
    object_directory = root / "objects" / digest[:2]
    object_directory.mkdir(parents=True, exist_ok=True)
    destination = object_directory / digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=".object-", dir=object_directory)
    temporary = Path(temporary_name)
    try:
        copied_digest = hashlib.sha256()
        copied_size = 0
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            while chunk := input_stream.read(1024 * 1024):
                copied_digest.update(chunk)
                copied_size += len(chunk)
                output.write(chunk)
            if copied_digest.hexdigest() != digest or copied_size != size:
                raise ValueError("build output changed while publishing its content object")
            os.fchmod(output.fileno(), 0o444)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
            _fsync_directory(object_directory)
        except FileExistsError:
            actual, actual_size = _file_digest(destination, require_single_link=True)
            if actual != digest or actual_size != size:
                raise ValueError("content-addressed object collision or corruption")
    finally:
        temporary.unlink(missing_ok=True)
    return digest, size


def _publish_manifest(root: Path, spec: BuildNodeSpec, artifacts: list[dict[str, Any]]) -> Path:
    node_directory = root / "nodes"
    node_directory.mkdir(parents=True, exist_ok=True)
    path = node_directory / f"{spec.node_key}.json"
    payload = _canonical_json(
        {
            "schema": _NODE_SCHEMA,
            "node_key": spec.node_key,
            "identity": spec.identity_payload,
            "artifacts": artifacts,
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=".node-", dir=node_directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            os.fchmod(output.fileno(), 0o444)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(node_directory)
        except FileExistsError:
            if _stable_bytes(path, maximum_bytes=16 * 1024 * 1024) != payload:
                raise ValueError("immutable build-node key already has a different manifest")
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run_build_node(
    root: str | Path,
    spec: BuildNodeSpec,
    build: Callable[[Path], None],
) -> BuildNodeResult:
    """Run once or reuse a verified node; publish output objects before its manifest.

    ``build`` receives a fresh temporary directory and must write exactly the
    output roster declared by ``spec``. A failed build publishes no node
    manifest; retrying the same spec is safe and cannot replace existing data.
    """
    if not isinstance(spec, BuildNodeSpec):
        raise TypeError("spec must be a BuildNodeSpec")
    if not callable(build):
        raise TypeError("build must be callable")
    store_root = Path(root).expanduser().resolve()
    store_root.mkdir(parents=True, exist_ok=True)
    _verify_input_files(spec)
    for dependency_name, dependency_key in spec.dependencies.items():
        try:
            _read_node_manifest(store_root, dependency_key)
        except FileNotFoundError as error:
            raise ValueError(
                f"build-node dependency {dependency_name!r} is not present in the store"
            ) from error
    existing = _load_existing_node(store_root, spec)
    if existing is not None:
        return existing
    staging_root = store_root / ".staging"
    staging_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{spec.node_key}-", dir=staging_root) as directory:
        output_root = Path(directory)
        build(output_root)
        observed: list[str] = []
        artifacts: list[dict[str, Any]] = []
        for path in output_root.rglob("*"):
            if path.is_symlink():
                raise ValueError("build-node outputs cannot contain symlinks")
            if path.is_file():
                relative = path.relative_to(output_root).as_posix()
                _relative_output(relative)
                observed.append(relative)
        if tuple(sorted(observed)) != spec.outputs:
            raise ValueError(
                f"build-node output roster differs from declaration: expected {spec.outputs}, "
                f"observed {tuple(sorted(observed))}"
            )
        for relative in spec.outputs:
            digest, size = _publish_object(store_root, output_root / relative)
            artifacts.append({"path": relative, "sha256": digest, "size_bytes": size})
        manifest_path = _publish_manifest(store_root, spec, artifacts)
    result = _load_existing_node(store_root, spec)
    if result is None or result.manifest_path != manifest_path:
        raise RuntimeError("build-node manifest publication could not be verified")
    return BuildNodeResult(result.node_key, result.manifest_path, result.artifacts, False)


def resolve_build_artifact(root: str | Path, result: BuildNodeResult, output_path: str) -> Path:
    """Resolve a declared output name to its verified immutable content object."""
    relative = _relative_output(output_path).as_posix()
    artifact = next((row for row in result.artifacts if row["path"] == relative), None)
    if artifact is None:
        raise KeyError(f"build node has no declared output {relative!r}")
    path = (
        Path(root).expanduser().resolve() / "objects" / artifact["sha256"][:2] / artifact["sha256"]
    )
    digest, size = _file_digest(path, require_single_link=True)
    if digest != artifact["sha256"] or size != artifact["size_bytes"]:
        raise ValueError("resolved build artifact failed integrity verification")
    return path

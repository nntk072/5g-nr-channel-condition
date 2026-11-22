"""Fail-closed resource authorization and bounded native-resource accounting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import jsonschema
from artifacts.provenance import atomic_json

from .contracts import (
    ROOT,
    ContractError,
    canonical_json_bytes,
    load_schema,
    load_strict_json_bytes,
)

LOCAL_CLASSES = {"docs_only", "cpu_test"}
NATIVE_CLASSES = {"native_smoke", "scientific_native", "production_d1"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTED_DEVICE = re.compile(r"^(?:cpu|cuda:(?:0|[1-9][0-9]*))$")
_USAGE_SCHEMA = "nr-agentops.resource-usage.v1"


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError("resource authorization expiry must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ContractError("resource authorization expiry must include a timezone")
    return parsed.astimezone(timezone.utc)


def _gpu_inventory() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            shell=False,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            memory_mib = int(parts[2])
        except ValueError:
            continue
        devices.append({"index": index, "name": parts[1], "memory_mib": memory_mib})
    return devices


def host_inventory(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    ram_bytes: int | None = None
    if hasattr(os, "sysconf"):
        try:
            ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            ram_bytes = None
    packages: dict[str, str] = {}
    for name in ("sionna-no-rt", "sionna-rt", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "disk_free_bytes": usage.free,
        "packages": packages,
        "gpu_devices": _gpu_inventory(),
    }


def _read_stable_file(path: Path, *, maximum_bytes: int = 64 * 1024 * 1024) -> bytes:
    try:
        path_stat = os.lstat(path)
    except OSError as error:
        raise ContractError("cannot read measured resource-budget report") from error
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
        raise ContractError("resource-budget report must be a single-link regular file")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError("cannot safely open measured resource-budget report") from error
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum_bytes or (before.st_dev, before.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise ContractError("resource-budget report exceeds its byte bound or changed on open")
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

    def identity(row: os.stat_result) -> tuple[int, ...]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
            row.st_nlink,
        )

    if (
        len(chunks) > maximum_bytes
        or identity(before) != identity(after)
        or identity(before) != identity(path_after)
    ):
        raise ContractError("resource-budget report changed during bounded read")
    return bytes(chunks)


def validate_budget_report_binding(authorization: dict[str, Any], report_path: Path) -> str:
    raw = _read_stable_file(report_path)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != authorization["budget_report_sha256"]:
        raise ContractError("resource authorization budget-report digest does not match the report")

    value = load_strict_json_bytes(raw)
    if not isinstance(value, dict):
        raise ContractError("resource-budget report must contain one JSON object")
    schema_path = ROOT / "compute/schema/resource-budget-report.v1.schema.json"
    try:
        schema_value = load_strict_json_bytes(schema_path.read_bytes())
    except OSError as error:
        raise ContractError("resource-budget report schema is unavailable") from error
    if not isinstance(schema_value, dict):
        raise ContractError("resource-budget report schema must be one JSON object")
    errors = list(jsonschema.Draft202012Validator(schema_value).iter_errors(value))
    if errors:
        raise ContractError(
            "resource-budget report violates schema: "
            + "; ".join(error.message for error in errors[:8])
        )
    return digest


def validate_scheduler_argv(
    authorization: dict[str, Any],
    scheduler_argv: Sequence[str],
) -> tuple[str, ...]:
    actual = tuple(scheduler_argv)
    if not actual or not all(isinstance(item, str) and item for item in actual):
        raise ContractError("scheduler argv must contain nonempty string arguments")
    allowed = tuple(authorization["allowed_scheduler_argv"])
    if actual != allowed:
        raise ContractError("scheduler argv does not exactly match the resource authorization")
    return actual


def validate_resource_authorization(
    authorization: dict[str, Any],
    *,
    task_id: str,
    source_revision: str,
    task_card_sha256: str,
    resource_class: str,
    host: str,
    budget_report_path: Path | None = None,
    now: datetime | None = None,
    inventory: dict[str, Any] | None = None,
) -> str:
    schema = load_schema("resource-authorization.v1.schema.json")
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(authorization))
    if errors:
        raise ContractError(
            "resource authorization violates schema: "
            + "; ".join(error.message for error in errors[:8])
        )
    expected = {
        "task_id": task_id,
        "source_revision": source_revision,
        "task_card_sha256": task_card_sha256,
        "resource_class": resource_class,
        "host": host,
    }
    for key, value in expected.items():
        if authorization[key] != value:
            raise ContractError(f"resource authorization {key} does not match the task")
    reference = now or datetime.now(timezone.utc)
    if _parse_utc(authorization["expires_at_utc"]) <= reference.astimezone(timezone.utc):
        raise ContractError("resource authorization expired")
    if resource_class == "scientific_native":
        # The boolean is an operator-supplied claim, not a gate-issued receipt. No verifier
        # for scientific gate receipts exists in AgentOps, so admission fails closed.
        raise ContractError(
            "scientific_native remains denied: no gate-issued scientific authorization "
            "receipt verifier exists, and an AgentOps boolean cannot substitute for one"
        )
    if resource_class == "production_d1":
        raise ContractError(
            "production_d1 remains denied because the existing D1 production gate is disabled"
        )
    if resource_class in NATIVE_CLASSES:
        selected_device = authorization["selected_device"]
        if not _SELECTED_DEVICE.fullmatch(selected_device):
            raise ContractError("native selected device must be 'cpu' or canonical cuda:<index>")
        if budget_report_path is None:
            raise ContractError("native resource authorization requires a measured budget report")
        validate_budget_report_binding(authorization, budget_report_path)
    if inventory is not None:
        if inventory.get("host") != host:
            raise ContractError("measured host inventory does not match the authorized host")
        ram_bytes = inventory.get("ram_bytes")
        if isinstance(ram_bytes, int) and authorization["ram_ceiling_bytes"] > ram_bytes:
            raise ContractError("resource authorization exceeds measured RAM capacity")
        disk_bytes = inventory.get("disk_free_bytes")
        if isinstance(disk_bytes, int) and authorization["disk_ceiling_bytes"] > disk_bytes:
            raise ContractError("resource authorization exceeds measured free disk capacity")
        packages = inventory.get("packages")
        if resource_class in NATIVE_CLASSES:
            if not isinstance(packages, dict):
                raise ContractError(
                    "native resource authorization requires measured package inventory"
                )
            sionna_versions = [packages.get("sionna-no-rt"), packages.get("sionna-rt")]
            if not any(
                isinstance(version, str) and version.startswith("2.2")
                for version in sionna_versions
            ):
                raise ContractError("native resource authorization requires Sionna 2.2 runtime")
            selected_device = authorization["selected_device"]
            if selected_device.startswith("cuda:"):
                try:
                    selected_index = int(selected_device.split(":", 1)[1])
                except ValueError as error:
                    raise ContractError("selected CUDA device must use cuda:<index>") from error
                devices = inventory.get("gpu_devices")
                if not isinstance(devices, list) or not any(
                    isinstance(device, dict) and device.get("index") == selected_index
                    for device in devices
                ):
                    raise ContractError(
                        "authorized CUDA device is unavailable on the measured host"
                    )
    return hashlib.sha256(canonical_json_bytes(authorization)).hexdigest()


def authorize_resource_class(
    resource_class: str,
    *,
    authorization: dict[str, Any] | None = None,
    task_id: str = "",
    source_revision: str = "",
    task_card_sha256: str = "",
    host: str = "",
    budget_report_path: Path | None = None,
    inventory: dict[str, Any] | None = None,
) -> str | None:
    if resource_class in LOCAL_CLASSES:
        if authorization is not None or budget_report_path is not None:
            raise ContractError("local resource classes do not accept native authorization inputs")
        return None
    if resource_class not in NATIVE_CLASSES:
        raise ContractError(f"unknown resource class {resource_class}")
    if authorization is None:
        raise ContractError(f"{resource_class} requires an exact resource authorization receipt")
    measured = inventory if inventory is not None else host_inventory(Path.cwd())
    return validate_resource_authorization(
        authorization,
        task_id=task_id,
        source_revision=source_revision,
        task_card_sha256=task_card_sha256,
        resource_class=resource_class,
        host=host,
        budget_report_path=budget_report_path,
        inventory=measured,
    )


class ResourceUsageLedger:
    """Durable, fail-closed cumulative GPU-hour accounting per authorization digest."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _paths(self, authorization_digest: str) -> tuple[Path, Path]:
        if not _SHA256.fullmatch(authorization_digest):
            raise ContractError("resource authorization digest must be lowercase SHA-256 hex")
        state = self.root / f"{authorization_digest}.json"
        lock = self.root / f"{authorization_digest}.lock"
        return state, lock

    @contextmanager
    def _lock(self, authorization_digest: str) -> Iterator[Path]:
        state_path, lock_path = self._paths(authorization_digest)
        self.root.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except FileExistsError as error:
            raise ContractError("resource usage ledger is locked for this authorization") from error
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.fsync(descriptor)
            os.close(descriptor)
            yield state_path
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _validate_hours(value: object, label: str, *, allow_zero: bool = True) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ContractError(f"{label} must be a numeric GPU-hour value")
        hours = float(value)
        if not math.isfinite(hours) or hours < 0 or (not allow_zero and hours == 0):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ContractError(f"{label} must be a finite {qualifier} GPU-hour value")
        return hours

    def _load_or_create(
        self,
        path: Path,
        authorization_digest: str,
        max_gpu_hours: float,
    ) -> dict[str, Any]:
        maximum = self._validate_hours(max_gpu_hours, "max_gpu_hours")
        if not path.exists():
            return {
                "schema": _USAGE_SCHEMA,
                "authorization_digest": authorization_digest,
                "max_gpu_hours": maximum,
                "entries": [],
            }
        value = load_strict_json_bytes(path.read_bytes())
        if not isinstance(value, dict):
            raise ContractError("resource usage ledger must contain one JSON object")
        if (
            value.get("schema") != _USAGE_SCHEMA
            or value.get("authorization_digest") != authorization_digest
            or not isinstance(value.get("entries"), list)
        ):
            raise ContractError("resource usage ledger has invalid identity or structure")
        stored_maximum = value.get("max_gpu_hours")
        stored_maximum = self._validate_hours(stored_maximum, "ledger max_gpu_hours")
        if not math.isclose(stored_maximum, maximum, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError("resource usage ledger max_gpu_hours changed")
        reservation_ids: set[str] = set()
        for entry in value["entries"]:
            if not isinstance(entry, dict):
                raise ContractError("resource usage ledger entry must be an object")
            required = {"reservation_id", "requested_gpu_hours", "status", "actual_gpu_hours"}
            if not required.issubset(entry):
                raise ContractError("resource usage ledger entry is incomplete")
            reservation_id = entry["reservation_id"]
            if (
                not isinstance(reservation_id, str)
                or not reservation_id
                or reservation_id.strip() != reservation_id
            ):
                raise ContractError("resource usage ledger reservation ID is invalid")
            if reservation_id in reservation_ids:
                raise ContractError("resource usage ledger has duplicate reservation IDs")
            reservation_ids.add(reservation_id)
            self._validate_hours(
                entry["requested_gpu_hours"],
                "ledger requested_gpu_hours",
                allow_zero=False,
            )
            status = entry["status"]
            if status not in {"pending", "completed", "failed"}:
                raise ContractError("resource usage ledger entry has invalid status")
            actual = entry["actual_gpu_hours"]
            if status == "pending":
                if actual is not None:
                    raise ContractError(
                        "pending resource usage entry must not have actual GPU hours"
                    )
            else:
                self._validate_hours(actual, "ledger actual_gpu_hours")
        return value

    @staticmethod
    def _totals(state: dict[str, Any]) -> tuple[float, float]:
        consumed = 0.0
        pending = 0.0
        for entry in state["entries"]:
            requested = ResourceUsageLedger._validate_hours(
                entry["requested_gpu_hours"],
                "ledger requested_gpu_hours",
                allow_zero=False,
            )
            actual = entry["actual_gpu_hours"]
            if entry["status"] == "pending":
                if actual is not None:
                    raise ContractError(
                        "pending resource usage entry must not have actual GPU hours"
                    )
                pending += requested
            else:
                consumed += ResourceUsageLedger._validate_hours(
                    actual,
                    "ledger actual_gpu_hours",
                )
        return consumed, pending

    def remaining_gpu_hours(self, authorization_digest: str, max_gpu_hours: float) -> float:
        state_path, _ = self._paths(authorization_digest)
        state = self._load_or_create(state_path, authorization_digest, max_gpu_hours)
        consumed, pending = self._totals(state)
        return max(0.0, float(max_gpu_hours) - consumed - pending)

    def reserve(
        self,
        authorization_digest: str,
        max_gpu_hours: float,
        *,
        reservation_id: str,
        requested_gpu_hours: float,
    ) -> None:
        requested = self._validate_hours(
            requested_gpu_hours,
            "requested_gpu_hours",
            allow_zero=False,
        )
        if not reservation_id or reservation_id.strip() != reservation_id:
            raise ContractError("resource reservation ID must be a nonempty trimmed string")
        with self._lock(authorization_digest) as state_path:
            state = self._load_or_create(state_path, authorization_digest, max_gpu_hours)
            if any(entry["reservation_id"] == reservation_id for entry in state["entries"]):
                raise ContractError("resource reservation ID has already been used")
            consumed, pending = self._totals(state)
            maximum = float(max_gpu_hours)
            if consumed + pending + requested > maximum + 1e-12:
                raise ContractError("requested GPU hours exceed the authorization ceiling")
            state["entries"].append(
                {
                    "reservation_id": reservation_id,
                    "requested_gpu_hours": requested,
                    "status": "pending",
                    "actual_gpu_hours": None,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "ended_at_utc": None,
                }
            )
            atomic_json(state_path, state)

    def settle(
        self,
        authorization_digest: str,
        max_gpu_hours: float,
        *,
        reservation_id: str,
        actual_gpu_hours: float,
        outcome: str,
    ) -> None:
        actual = self._validate_hours(actual_gpu_hours, "actual_gpu_hours")
        if outcome not in {"completed", "failed"}:
            raise ContractError("resource usage outcome must be completed or failed")
        with self._lock(authorization_digest) as state_path:
            state = self._load_or_create(state_path, authorization_digest, max_gpu_hours)
            match = None
            for entry in state["entries"]:
                if entry["reservation_id"] == reservation_id:
                    match = entry
                    break
            if match is None:
                raise ContractError("resource reservation ID does not exist")
            if match["status"] != "pending":
                raise ContractError("resource reservation has already been settled")
            consumed, _ = self._totals(state)
            if consumed + actual > float(max_gpu_hours) + 1e-12:
                raise ContractError("actual GPU usage exceeds the authorization ceiling")
            match["status"] = outcome
            match["actual_gpu_hours"] = actual
            match["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)


@contextmanager
def metered_gpu_execution(
    ledger: ResourceUsageLedger,
    authorization_digest: str,
    authorization: dict[str, Any],
    *,
    maximum_timeout_seconds: int,
) -> Iterator[int]:
    """Reserve a bounded CUDA execution window and charge elapsed time on every outcome."""

    if not authorization["selected_device"].startswith("cuda:"):
        yield maximum_timeout_seconds
        return

    remaining = ledger.remaining_gpu_hours(
        authorization_digest,
        float(authorization["max_gpu_hours"]),
    )
    available_seconds = math.floor(remaining * 3600.0)
    if available_seconds <= 1:
        raise ContractError("resource authorization has no GPU-hour budget left")
    timeout = min(maximum_timeout_seconds, available_seconds - 1)
    requested_hours = (timeout + 1) / 3600.0
    reservation_id = uuid.uuid4().hex
    ledger.reserve(
        authorization_digest,
        float(authorization["max_gpu_hours"]),
        reservation_id=reservation_id,
        requested_gpu_hours=requested_hours,
    )
    started = time.monotonic()
    try:
        yield timeout
    except BaseException:
        elapsed = (time.monotonic() - started) / 3600.0
        ledger.settle(
            authorization_digest,
            float(authorization["max_gpu_hours"]),
            reservation_id=reservation_id,
            actual_gpu_hours=elapsed,
            outcome="failed",
        )
        raise
    else:
        elapsed = (time.monotonic() - started) / 3600.0
        ledger.settle(
            authorization_digest,
            float(authorization["max_gpu_hours"]),
            reservation_id=reservation_id,
            actual_gpu_hours=elapsed,
            outcome="completed",
        )

"""Canonical paths for a single immutable experiment run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Typed paths below one caller-selected run directory."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "RunLayout":
        return cls(Path(root).expanduser().resolve())

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def complete(self) -> Path:
        return self.root / "COMPLETE"

    def ensure(self) -> None:
        if self.complete.exists():
            raise FileExistsError(f"Completed run is immutable: {self.root}")
        for path in (
            self.root,
            self.config,
            self.models,
            self.predictions,
            self.evaluation,
            self.diagnostics,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)

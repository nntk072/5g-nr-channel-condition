"""Check that every coverage exclusion has a reviewed reason in the coverage policy.

The coverage configuration in ``pyproject.toml`` may omit files and exclude line
patterns. Each entry must appear as a code span in the first column of a table
row of ``docs/architecture/coverage-policy.md`` whose second column gives the
reason. An unlisted entry, or a listed entry that is no longer configured, fails.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs" / "architecture" / "coverage-policy.md"
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(\S.*?)\s*\|\s*$", re.MULTILINE)


def configured_exclusions() -> set[str]:
    """Return every omit path and exclude pattern named in ``pyproject.toml``."""
    coverage = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["coverage"]
    entries = set(coverage.get("run", {}).get("omit", []))
    entries.update(coverage.get("report", {}).get("exclude_also", []))
    return entries


def documented_exclusions() -> dict[str, str]:
    """Return the reviewed entries and their reasons from the policy table."""
    return {match.group(1): match.group(2) for match in ROW.finditer(POLICY.read_text(encoding="utf-8"))}


def check() -> list[str]:
    """Return a list of problems; an empty list means the policy matches the configuration."""
    configured = configured_exclusions()
    documented = documented_exclusions()
    problems = [f"unlisted coverage exclusion: {entry}" for entry in sorted(configured - documented.keys())]
    problems += [f"documented exclusion no longer configured: {entry}" for entry in sorted(documented.keys() - configured)]
    problems += [f"empty reason for {entry}" for entry, reason in documented.items() if not reason or reason == "TODO"]
    return problems


def main(argv: list[str] | None = None) -> int:
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"coverage policy matches pyproject.toml ({len(configured_exclusions())} exclusions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

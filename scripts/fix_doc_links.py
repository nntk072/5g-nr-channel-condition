"""Repair broken markdown links in documents that may be edited, and report the rest.

A broken relative link is repaired when the recorded path moves in
``docs/reference/digest-rebindings.json`` lead to an existing file. Links in
frozen documents (any file under ``docs/research/`` that is not listed in
``EDITABLE``) are never rewritten; they are reported with the file and line.

Usage:
    python scripts/fix_doc_links.py            # repair editable documents, report the rest
    python scripts/fix_doc_links.py --check    # exit 1 if any broken link remains in editable documents
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REBINDINGS = ROOT / "docs" / "reference" / "digest-rebindings.json"
LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")
# The master plan is an authorised, edited document; other research files are frozen.
EDITABLE = {"docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md"}
SKIP_PREFIXES = ("http://", "https://", "mailto:")


def moves() -> dict[str, list[str]]:
    data = json.loads(REBINDINGS.read_text(encoding="utf-8"))
    table: dict[str, list[str]] = {}
    for entry in data.get("file_moves", []):
        table.setdefault(entry["old_path"], []).append(entry["new_path"])
    return table


def current_location(repo_path: str, table: dict[str, list[str]]) -> str | None:
    """Follow the recorded moves from ``repo_path`` to an existing file."""
    seen: set[str] = set()
    queue = [repo_path]
    while queue:
        candidate = queue.pop(0)
        if candidate in seen:
            continue
        seen.add(candidate)
        if (ROOT / candidate).exists():
            return candidate
        queue.extend(reversed(table.get(candidate, [])))
    return None


def base_research_digests() -> set[str]:
    """SHA-256 of every research document at the evidence base; these bytes are frozen."""
    import hashlib
    import subprocess

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "016168e", "docs/research"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    digests = set()
    for name in listing:
        data = subprocess.run(
            ["git", "show", f"016168e:{name}"], cwd=ROOT, capture_output=True
        ).stdout
        digests.add(hashlib.sha256(data).hexdigest())
    return digests


def is_frozen(path: Path, frozen_digests: set[str]) -> bool:
    import hashlib

    rel = path.relative_to(ROOT).as_posix()
    if rel in EDITABLE:
        return False
    if rel.startswith("docs/research/"):
        return True
    return hashlib.sha256(path.read_bytes()).hexdigest() in frozen_digests


FULL_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s#]+)(#[^)\s]*)?\)")


def unique_basename(name: str) -> str | None:
    """Return the repository path of the only file with this basename under docs/ or dev-docs/."""
    matches = []
    for base in ("docs", "dev-docs"):
        for path in (ROOT / base).rglob(name):
            if path.is_file() and "_build" not in path.parts and "generated" not in path.parts:
                matches.append(path.relative_to(ROOT).as_posix())
    return matches[0] if len(matches) == 1 else None


def documents() -> list[Path]:
    found = []
    for base in ("docs", "dev-docs"):
        for path in (ROOT / base).rglob("*.md"):
            if "_build" in path.parts or "generated" in path.parts:
                continue
            found.append(path)
    found.append(ROOT / "README.md")
    found.append(ROOT / "ARCHITECTURE.md")
    return sorted(found)


def process(
    path: Path, table: dict[str, list[str]], write: bool, frozen_digests: set[str]
) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    editable = not is_frozen(path, frozen_digests)
    out_lines = []
    changed = False
    for number, line in enumerate(text.split("\n"), start=1):

        def fix(match: re.Match[str]) -> str:
            nonlocal changed
            target = match.group(2)
            if target.startswith(SKIP_PREFIXES) or target.startswith("/"):
                return match.group(0)
            resolved = (path.parent / target).resolve()
            try:
                rel_target = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                return match.group(0)
            if resolved.exists():
                return match.group(0)
            new_repo = current_location(rel_target, table)
            if new_repo is not None and editable and write:
                changed = True
                new_rel = os.path.relpath(ROOT / new_repo, path.parent).replace(os.sep, "/")
                return f"{match.group(1)}{new_rel}{match.group(3) or ''}{match.group(4)}"
            if new_repo is None:
                base_name = Path(target).name
                found = unique_basename(base_name)
                if found is not None and editable and write:
                    changed = True
                    new_rel = os.path.relpath(ROOT / found, path.parent).replace(os.sep, "/")
                    return f"{match.group(1)}{new_rel}{match.group(3) or ''}{match.group(4)}"
                problems.append(f"{rel}:{number}: unresolved link {target}")
            else:
                problems.append(f"{rel}:{number}: link {target} moved to {new_repo}")
            return match.group(0)

        line = LINK.sub(fix, line)
        if editable and write:
            # A target that is not in this tree becomes plain text, marked as such.
            def plain(match: re.Match[str]) -> str:
                nonlocal changed
                target = match.group(2)
                if target.startswith(SKIP_PREFIXES) or target.startswith("/"):
                    return match.group(0)
                resolved = (path.parent / target).resolve()
                if resolved.exists() or not resolved.is_relative_to(ROOT):
                    return match.group(0)
                if current_location(resolved.relative_to(ROOT).as_posix(), table) is not None:
                    return match.group(0)
                if unique_basename(Path(target).name) is not None:
                    return match.group(0)
                changed = True
                return f"{match.group(1)} (not in this tree)"

            line = FULL_LINK.sub(
                lambda m: plain(m) if m.group(0).startswith("[") else m.group(0), line
            )
        out_lines.append(line)
    if changed:
        path.write_text("\n".join(out_lines), encoding="utf-8")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    table = moves()
    frozen_digests = base_research_digests()
    problems: list[str] = []
    for path in documents():
        problems.extend(process(path, table, write=not args.check, frozen_digests=frozen_digests))
    for item in problems:
        print(item)
    editable_left = [
        p for p in problems if not is_frozen(ROOT / p.split(":", 1)[0], frozen_digests)
    ]
    print(f"{len(problems)} link problem(s); {len(editable_left)} in editable documents")
    if args.check and editable_left:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

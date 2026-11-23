"""Check relative Markdown links under docs/ and README.md, and index reachability.

Reports every link whose target does not exist, and every document under docs/
that no index file links to. Exits non-zero when anything is reported.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK = re.compile(r"\]\(([^)\s]+)\)")
INDEXES = (
    DOCS / "README.md",
    DOCS / "research" / "INDEX.md",
    DOCS / "research" / "results" / "INDEX.md",
)


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md", *DOCS.rglob("*.md")]
    return [path for path in files if path.is_file()]


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#"))


def broken_links() -> list[str]:
    problems = []
    for path in _markdown_files():
        for match in LINK.finditer(path.read_text(encoding="utf-8", errors="replace")):
            target = match.group(1)
            if _is_external(target):
                continue
            target_path = unquote(target.split("#", 1)[0])
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                problems.append(f"{path.relative_to(ROOT).as_posix()}: broken link {target}")
    return problems


def unreachable_documents() -> list[str]:
    linked: set[Path] = set()
    for index in INDEXES:
        if not index.is_file():
            continue
        for match in LINK.finditer(index.read_text(encoding="utf-8")):
            target = unquote(match.group(1).split("#", 1)[0])
            linked.add((index.parent / target).resolve())
    problems = []
    for path in sorted(DOCS.rglob("*")):
        if not path.is_file() or path.name in {"README.md", "INDEX.md"}:
            continue
        if path.suffix not in {".md", ".json", ".pdf", ".png", ".svg", ".gz", ".sha256"}:
            continue
        if path.resolve() not in linked:
            problems.append(f"not reachable from an index: {path.relative_to(ROOT).as_posix()}")
    return problems


def main() -> int:
    problems = broken_links() + unreachable_documents()
    for line in problems:
        print(line)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

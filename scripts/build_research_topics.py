"""Build the research topic notes from the archived master plan at its git tag.

The archived plan is read with ``git show``; the repository tree is not
changed by the archive. Each topic note copies whole level-2 sections of the
archived plan verbatim and records the source path and line range of every
copied section. Run ``python scripts/build_research_topics.py`` to rebuild the
notes, or ``--check`` to verify that they match the archive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_TAG = "docs/master-research-plan-archive"
ARCHIVE_PATH = "docs/research/MASTER_RESEARCH_PLAN.md"
OUTPUT_DIR = ROOT / "docs" / "research"

TOPICS: dict[str, tuple[str, list[str]]] = {
    "PLAN_ARCHITECTURE.md": (
        "Architecture",
        [
            "## 1. Current-repository audit",
            "## 4. Sionna RT 2.2",
            "## Appendix B.",
            "## Appendix C.",
        ],
    ),
    "PLAN_DATA_GENERATION.md": (
        "Data generation",
        ["## 5. Data program"],
    ),
    "PLAN_RECEIVERS_AND_TARGETS.md": (
        "Receivers, targets and methods",
        [
            "## 2. Core scientific definitions",
            "## 3. Full output inventory",
            "## 6. The paper's actual methodological",
        ],
    ),
    "PLAN_QUALIFICATION_GATES.md": (
        "Qualification gates and evidence",
        [
            "## 7. Complete experiment catalogue",
            "## 9. Validation, CI",
            "## 15. Final project charter",
            "## Appendix E.",
        ],
    ),
    "PLAN_OPEN_QUESTIONS.md": (
        "Open questions and risks",
        [
            "## 0. Executive decisions",
            "## 11. Risk register",
            "## 12. Manuscript outline",
            "## 13. Practical execution checklist",
        ],
    ),
}


def archive_lines() -> list[str]:
    text = subprocess.run(
        ["git", "show", f"{ARCHIVE_TAG}:{ARCHIVE_PATH}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return text.split("\n")


def sections(lines: list[str], prefixes: list[str]) -> list[tuple[str, int, int, list[str]]]:
    """Return (heading, first line, last line, body) for each matching level-2 section."""
    found = []
    for prefix in prefixes:
        starts = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(starts) != 1:
            raise SystemExit(f"heading prefix {prefix!r} matched {len(starts)} times")
        start = starts[0]
        end = next(
            (
                j
                for j in range(start + 1, len(lines))
                if lines[j].startswith("## ") or lines[j].startswith("# ")
            ),
            len(lines),
        )
        found.append((lines[start], start + 1, end, lines[start:end]))
    return found


def build_topic(title: str, prefixes: list[str], lines: list[str], file_name: str) -> str:
    out = [
        f"# Research plan topic: {title.lower()}",
        "",
        "This note is built from the archived master research plan at git tag",
        f"`{ARCHIVE_TAG}`, file `{ARCHIVE_PATH}`. Each section below is copied",
        "verbatim; the source line range is given before it. The archived plan is",
        "not restored to the tree. Rebuild this note with",
        "`python scripts/build_research_topics.py`.",
        "",
    ]
    for heading, first, last, body in sections(lines, prefixes):
        out.append(f"<!-- archive {ARCHIVE_PATH} lines {first}-{last} -->")
        out.extend(body)
        while out and out[-1] == "":
            out.pop()
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the notes differ from a fresh build"
    )
    args = parser.parse_args(argv)
    lines = archive_lines()
    stale = []
    for file_name, (title, prefixes) in TOPICS.items():
        text = build_topic(title, prefixes, lines, file_name)
        path = OUTPUT_DIR / file_name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(file_name)
        else:
            path.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print("stale topic notes:", ", ".join(stale))
            return 1
        print("topic notes match the archive")
        return 0
    print("topic notes written:", ", ".join(TOPICS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

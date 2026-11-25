"""Derive every package-list setting from config/packages.toml.

Rewrites the package-list blocks of pyproject.toml. Run with --check to fail when
pyproject.toml is out of date.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config" / "packages.toml"
PYPROJECT = ROOT / "pyproject.toml"


def load_packages(path: Path = SOURCE) -> dict[str, list[str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))["packages"]
    return {
        "names": list(data["names"]),
        "test_roots": list(data["test_roots"]),
    }


def _array(items: list[str], indent: str) -> str:
    body = "".join(f'{indent}  "{item}",\n' for item in items)
    return "[\n" + body + indent + "]"


def render(text: str, packages: dict[str, list[str]]) -> str:
    names = packages["names"]
    coverage = list(names)

    def replace_block(source: str, section: str, key: str, value: list[str], inline: bool) -> str:
        header = re.escape(f"[{section}]")
        pattern = re.compile(rf"(?ms)({header}\n.*?^{key} = )(\[.*?\])", re.M)
        if inline:
            rendered = "[" + ", ".join(f'"{item}"' for item in value) + "]"
        else:
            rendered = _array(value, "")
        new, count = pattern.subn(lambda m: m.group(1) + rendered, source, count=1)
        if count != 1:
            raise SystemExit(f"pyproject.toml: section [{section}] key {key} not found")
        return new

    setuptools_include = [f"{name}*" if name != "scripts" else "scripts*" for name in names]
    text = replace_block(
        text, "tool.setuptools.packages.find", "include", setuptools_include, False
    )
    text = replace_block(text, "tool.coverage.run", "source", coverage, True)
    text = replace_block(text, "tool.ruff", "src", names, True)
    text = replace_block(text, "tool.mypy", "files", names, False)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if pyproject.toml is stale")
    args = parser.parse_args(argv)
    current = PYPROJECT.read_text(encoding="utf-8")
    rendered = render(current, load_packages())
    if args.check:
        if rendered != current:
            print(
                "pyproject.toml package lists are stale; run tools/sync_packages.py",
                file=sys.stderr,
            )
            return 1
        print("pyproject.toml package lists are current")
        return 0
    PYPROJECT.write_text(rendered, encoding="utf-8")
    print("pyproject.toml package lists written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

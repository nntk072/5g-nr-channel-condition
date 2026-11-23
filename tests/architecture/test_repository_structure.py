"""Structure rules for the repository layout.

These tests fail when a placeholder file returns under data/, when a phase
token (a G-number or an edition name) enters a path, when runtime imports form a
cycle, or when a package directory has no README.md.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = tuple(tomllib.loads((ROOT / "config/packages.toml").read_text())["packages"]["names"])
PHASE_TOKEN = re.compile(r"(^|[/_.\-])[gG][0-9]+([._\-]|$)|edition", re.IGNORECASE)


def _tracked() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def _module_name(relative: str) -> str:
    parts = list(Path(relative).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def test_no_gitkeep_placeholders_under_data() -> None:
    kept = [path for path in _tracked() if path.startswith("data/") and path.endswith(".gitkeep")]
    on_disk = [p.relative_to(ROOT).as_posix() for p in (ROOT / "data").rglob(".gitkeep")]
    assert kept == []
    assert on_disk == []


def test_no_phase_token_in_tracked_paths() -> None:
    offending = [path for path in _tracked() if PHASE_TOKEN.search(path)]
    assert offending == [], offending


def test_runtime_imports_between_packages_form_no_cycle() -> None:
    files = [p for p in _tracked() if p.endswith(".py") and p.split("/")[0] in PACKAGES]
    modules = {_module_name(p): p for p in files}
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, relative in modules.items():
        is_package = relative.endswith("__init__.py")
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        # Module-level statements only: TYPE_CHECKING blocks and function bodies do not run
        # at import time, so they cannot create an import cycle.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = name.split(".") if is_package else name.split(".")[:-1]
                    if node.level > 1:
                        base = base[: len(base) - (node.level - 1)]
                    target = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    target = node.module or ""
                candidates = [target, *(f"{target}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            else:
                continue
            graph[name].update(c for c in candidates if c in modules and c != name)
    assert _cycles(graph) == []


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return strongly connected components with more than one module (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []
    counter = [0]

    def visit(node: str) -> None:
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for child in graph.get(node, ()):
            if child not in index:
                visit(child)
                low[node] = min(low[node], low[child])
            elif child in on_stack:
                low[node] = min(low[node], index[child])
        if low[node] == index[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in graph.get(node, ()):
                components.append(sorted(component))

    for node in graph:
        if node not in index:
            visit(node)
    return components


def test_every_package_directory_has_a_readme() -> None:
    directories: set[Path] = set()
    for relative in _tracked():
        if not relative.endswith(".py") or relative.split("/")[0] not in PACKAGES:
            continue
        parent = Path(relative).parent
        while parent.parts and parent.parts[0] in PACKAGES:
            directories.add(parent)
            parent = parent.parent
    missing = sorted(d.as_posix() for d in directories if not (ROOT / d / "README.md").is_file())
    assert missing == []

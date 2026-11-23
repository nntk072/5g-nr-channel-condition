"""Build a source inventory of exported Python helpers and direct callers.

This is an audit aid, not a claim that syntactic references are canonical routes.
Only explicit rows in HELPER_WIRING_MATRIX.md count as documented route evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("docs/research/HELPER_WIRING_INVENTORY.json")
SCHEMA = Path("docs/research/schema/helper-wiring-inventory.v1.schema.json")
WORKFLOW_SCHEMAS = (
    Path("datasets/schema/network-pusch-workflow-report.v1.schema.json"),
    Path("datasets/schema/network-grid-sinr-workflow-report.v1.schema.json"),
    Path("datasets/schema/network-transmitter-pusch-workflow-report.v1.schema.json"),
)
MATRIX = Path("dev-docs/reference/helper-wiring-matrix.md")
PLAN_SOURCES = (
    Path("docs/research/MASTER_RESEARCH_PLAN.md"),
    Path("docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md"),
    Path("docs/research/IMPLEMENTATION_BACKLOG.md"),
    Path("docs/research/EXECUTION_QUEUE.md"),
)
# The master research plan was removed from the tree. Its text is still read from
# the archive tag so the reference inventory keeps the same plan-reference coverage.
ARCHIVE_REVISION = "docs/master-research-plan-archive"
ARCHIVED_SOURCES = frozenset({Path("docs/research/MASTER_RESEARCH_PLAN.md")})
PACKAGES = (
    "datasets",
    "evaluation",
    "experiments",
    "integrations/sionna",
    "propagation",
    "receivers",
)
SCAN_ROOTS = (
    "datasets",
    "evaluation",
    "experiments",
    "integrations",
    "propagation",
    "receivers",
    "scripts",
    "loslab",
)
TEST_SCAN_ROOTS = ("tests",)
EXCLUDED_PARTS = {"tests", ".venv", "build", "dist", "__pycache__"}
CODE = re.compile(r"`([^`]+)`")
_NON_EXPORTABLE_GENERIC_NAMES = {"load", "save", "select", "complete", "metadata", "sha256"}


def _source_bytes(root: Path, source: Path | str) -> bytes:
    """Return a plan or source file's bytes from the tree, or from the archive tag."""
    source = Path(source)
    path = root / source
    if path.is_file():
        return path.read_bytes()
    if source in ARCHIVED_SOURCES:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{ARCHIVE_REVISION}:{source.as_posix()}"],
            capture_output=True,
            check=True,
        )
        return completed.stdout
    raise FileNotFoundError(source.as_posix())


def _relative_module(base: Path, level: int, module: str | None) -> Path:
    parts = base.parts[: len(base.parts) - max(level - 1, 0)]
    return Path(*parts, *(module or "").split("."))


def _defining_function(
    module: Path, name: str, depth: int = 0
) -> tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Follow re-exports until the module that defines ``name`` is found."""
    if depth > 8:
        return None
    source_file = module.with_suffix(".py")
    if not (ROOT / source_file).is_file():
        source_file = module / "__init__.py"
    if not (ROOT / source_file).is_file():
        return None
    tree = ast.parse(
        (ROOT / source_file).read_text(encoding="utf-8"), filename=source_file.as_posix()
    )
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return source_file, node
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            target = (
                _relative_module(source_file.parent, node.level, node.module)
                if node.level
                else Path(*(node.module or "").split("."))
            )
            found = _defining_function(target, alias.name, depth + 1)
            if found is not None:
                return found
    return None


def _package_exports(package: str) -> list[dict[str, Any]]:
    base = Path(package)
    init_path = base / "__init__.py"
    tree = ast.parse((ROOT / init_path).read_text(encoding="utf-8"), filename=init_path.as_posix())
    exports: list[str] = []
    import_map: dict[str, tuple[Path, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                exports = value
        elif isinstance(node, ast.ImportFrom):
            source_module = (
                _relative_module(base, node.level, node.module)
                if node.level
                else Path(*(node.module or "").split("."))
            )
            for alias in node.names:
                import_map[alias.asname or alias.name] = (source_module, alias.name)

    results: list[dict[str, Any]] = []
    for public_name in exports:
        resolved = import_map.get(public_name)
        if resolved is None:
            continue
        source_module, source_name = resolved
        found = _defining_function(source_module, source_name)
        if found is None:
            continue
        source_file, node = found
        results.append(
            {
                "public_name": public_name,
                "qualified_name": f"{package.replace('/', '.')}.{public_name}",
                "source": source_file.as_posix(),
                "definition_line": node.lineno,
            }
        )
    return results


def _reference_index(
    exports: list[dict[str, Any]], *, scan_roots: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """Index directly imported function calls in one repository AST pass."""
    import_targets: dict[tuple[str, str], set[str]] = {}
    module_targets: dict[str, dict[str, set[str]]] = {}
    local_targets: dict[str, dict[str, set[str]]] = {}
    for export in exports:
        source_module = export["source"].removesuffix(".py").replace("/", ".")
        public_package = export["qualified_name"].rpartition(".")[0]
        local_targets.setdefault(export["source"], {}).setdefault(export["public_name"], set()).add(
            export["qualified_name"]
        )
        # A defining submodule inside a package is also reached through that package.
        source_path = Path(export["source"])
        defining_package = (
            source_path.parent.as_posix().replace("/", ".")
            if source_path.name != "__init__.py"
            and (ROOT / source_path.parent / "__init__.py").is_file()
            else None
        )
        for module, symbol in (
            (source_module, export["public_name"]),
            (public_package, export["public_name"]),
            *(((defining_package, export["public_name"]),) if defining_package else ()),
        ):
            import_targets.setdefault((module, symbol), set()).add(export["qualified_name"])
            module_targets.setdefault(module, {}).setdefault(symbol, set()).add(
                export["qualified_name"]
            )

    refs: dict[str, list[dict[str, Any]]] = {item["qualified_name"]: [] for item in exports}
    source_paths: list[Path] = []
    for scan_root in scan_roots:
        start = ROOT / scan_root
        if not start.exists():
            continue
        for directory, child_dirs, filenames in os.walk(start):
            child_dirs[:] = sorted(name for name in child_dirs if name not in EXCLUDED_PARTS)
            source_paths.extend(
                Path(directory) / filename for filename in filenames if filename.endswith(".py")
            )

    # A facade module re-exports names defined in a submodule. A caller that imports the
    # name from the facade reaches the defining submodule, so index that edge too.
    facade_paths: list[Path] = []
    for scan_root in (*SCAN_ROOTS, *TEST_SCAN_ROOTS):
        start = ROOT / scan_root
        if start.exists():
            facade_paths.extend(sorted(start.rglob("*.py")))
    for path in sorted(set(facade_paths)):
        if path.name == "__init__.py" or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        try:
            facade_tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        facade_module = path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
        for node in facade_tree.body:
            if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                continue
            for alias in node.names:
                found = _defining_function(Path(*node.module.split(".")), alias.name)
                if found is None:
                    continue
                defining_file = found[0]
                qualified = (
                    defining_file.with_suffix("").as_posix().replace("/", ".") + "." + alias.name
                )
                targets = {qualified}
                for export in exports:
                    if (
                        export["public_name"] == alias.name
                        and export["source"] == defining_file.as_posix()
                    ):
                        targets.add(export["qualified_name"])
                import_targets.setdefault((facade_module, alias.name), set()).update(targets)
                module_targets.setdefault(facade_module, {}).setdefault(alias.name, set()).update(
                    targets
                )

    for path in sorted(source_paths):
        relative = path.relative_to(ROOT)
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative.as_posix())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        local_aliases: dict[str, set[str]] = {
            name: set(targets)
            for name, targets in local_targets.get(relative.as_posix(), {}).items()
        }
        module_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_module = node.module or ""
                if node.level:
                    # Relative imports are resolved from the importing file's package.
                    package = relative.parent
                    imported_module = (
                        _relative_module(package, node.level, node.module)
                        .as_posix()
                        .replace("/", ".")
                    )
                for alias in node.names:
                    local = alias.asname or alias.name
                    targets = import_targets.get((imported_module, alias.name))
                    if targets:
                        local_aliases.setdefault(local, set()).update(targets)
                    # ``from package import submodule as alias`` is a module
                    # import, not a function import. Resolve it when the
                    # submodule defines one of the inventoried exports.
                    submodule = f"{imported_module}.{alias.name}".strip(".")
                    if submodule in module_targets:
                        module_aliases[local] = submodule
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_targets:
                        module_aliases[alias.asname or alias.name.split(".")[0]] = alias.name

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                targets = local_aliases.get(fn.id, set()) if isinstance(fn, ast.Name) else set()
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                    targets.update(
                        module_targets.get(module_aliases.get(fn.value.id, ""), {}).get(
                            fn.attr, set()
                        )
                    )
                for target in targets:
                    refs.setdefault(target, []).append(
                        {"path": relative.as_posix(), "line": node.lineno}
                    )
    return refs


def _non_exported_plan_named_definitions(
    *, root: Path, exports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Find source definitions whose exact snake-case names occur in plan code spans.

    Generic one-word method names and package-exported functions are excluded.
    This reports source locations only; a match is not evidence that the plan
    reference names the same semantic operation.
    """
    plan_references: dict[str, set[str]] = {}
    for source in PLAN_SOURCES:
        text = _source_bytes(root, source).decode("utf-8")
        for value in CODE.findall(text):
            symbol = value.rsplit(".", 1)[-1]
            if (
                symbol.isidentifier()
                and "_" in symbol.strip("_")
                and symbol.strip("_") not in _NON_EXPORTABLE_GENERIC_NAMES
            ):
                plan_references.setdefault(symbol, set()).add(source.as_posix())

    exported_definitions = {(item["source"], item["public_name"]) for item in exports}
    definitions: list[dict[str, Any]] = []
    for package in PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            relative = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            parents = {
                child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
            }
            module = relative.removesuffix(".py").replace("/", ".")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if (
                    node.name not in plan_references
                    or (relative, node.name) in exported_definitions
                ):
                    continue
                parent = parents.get(node)
                kind = "method" if isinstance(parent, ast.ClassDef) else "function"
                definitions.append(
                    {
                        "public_name": node.name,
                        "qualified_name": f"{module}.{node.name}",
                        "source": relative,
                        "definition_line": node.lineno,
                        "definition_kind": kind,
                        "plan_reference_files": sorted(plan_references[node.name]),
                    }
                )
    return sorted(
        definitions,
        key=lambda item: (item["qualified_name"], item["source"], item["definition_line"]),
    )


def build_inventory(root: Path = ROOT) -> dict[str, Any]:
    global ROOT
    previous_root = ROOT
    ROOT = root.resolve()
    try:
        matrix_text = (ROOT / MATRIX).read_text(encoding="utf-8")
        documented_names = {
            token.rsplit(".", 1)[-1]
            for token in CODE.findall(matrix_text.split("## Remaining source audit")[0])
            if token and " " not in token
        }
        exports: list[dict[str, Any]] = []
        for package in PACKAGES:
            exports.extend(_package_exports(package))
        production_ref_index = _reference_index(exports, scan_roots=SCAN_ROOTS)
        test_ref_index = _reference_index(exports, scan_roots=TEST_SCAN_ROOTS)
        items: list[dict[str, Any]] = []
        for export in exports:
            refs = production_ref_index[export["qualified_name"]]
            test_refs = test_ref_index[export["qualified_name"]]
            documented = export["public_name"] in documented_names
            items.append(
                {
                    **export,
                    "direct_production_references": refs,
                    "direct_test_references": test_refs,
                    "documented_route": documented,
                    "audit_disposition": (
                        "documented_route_requires_manual_review"
                        if documented
                        else "route_mapping_required"
                        if refs
                        else "no_direct_production_reference_found"
                    ),
                }
            )
        items.sort(key=lambda row: row["qualified_name"])
        plan_symbols = {
            source.as_posix(): set(
                re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]*",
                    _source_bytes(ROOT, source).decode("utf-8"),
                )
            )
            for source in PLAN_SOURCES
        }
        for item in items:
            item["exact_plan_reference_files"] = [
                source for source, symbols in plan_symbols.items() if item["public_name"] in symbols
            ]
        non_exported_plan_definitions = _non_exported_plan_named_definitions(
            root=ROOT, exports=exports
        )
        non_exported_refs = _reference_index(non_exported_plan_definitions, scan_roots=SCAN_ROOTS)
        non_exported_test_refs = _reference_index(
            non_exported_plan_definitions, scan_roots=TEST_SCAN_ROOTS
        )
        for item in non_exported_plan_definitions:
            item["direct_production_references"] = non_exported_refs[item["qualified_name"]]
            item["direct_test_references"] = non_exported_test_refs[item["qualified_name"]]
        source_paths = {
            MATRIX.as_posix(),
            "scripts/audit_helper_wiring.py",
            "pyproject.toml",
            SCHEMA.as_posix(),
            *(source.as_posix() for source in WORKFLOW_SCHEMAS if (ROOT / source).is_file()),
            *(source.as_posix() for source in PLAN_SOURCES),
            *(item["source"] for item in items),
            *(item["source"] for item in non_exported_plan_definitions),
            *(
                reference["path"]
                for item in non_exported_plan_definitions
                for reference in (
                    *item["direct_production_references"],
                    *item["direct_test_references"],
                )
            ),
            *(
                reference["path"]
                for item in items
                for reference in (
                    *item["direct_production_references"],
                    *item["direct_test_references"],
                )
            ),
        }
        inventory = {
            "schema": "loslab.helper_wiring_inventory.v1",
            "method": "AST inventory of package __all__ functions, direct production/test call references, and non-exported plan-named definitions; no semantic route inference",
            "scope_packages": list(PACKAGES),
            "reference_scan_roots": list(SCAN_ROOTS),
            "test_reference_scan_roots": list(TEST_SCAN_ROOTS),
            "source_sha256": {
                source: hashlib.sha256(_source_bytes(ROOT, source)).hexdigest()
                for source in sorted(source_paths)
            },
            "helper_count": len(items),
            "helpers": items,
            "non_exported_plan_named_definition_count": len(non_exported_plan_definitions),
            "non_exported_plan_named_definitions": non_exported_plan_definitions,
        }
        schema = json.loads((ROOT / SCHEMA).read_text(encoding="utf-8"))
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover - project dependency
            raise RuntimeError("jsonschema is required to validate the helper inventory") from error
        jsonschema.Draft202012Validator(schema).validate(inventory)
        return inventory
    finally:
        ROOT = previous_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    inventory = build_inventory()
    rendered = json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    path = ROOT / args.output
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"helper wiring inventory is missing or stale: {args.output}")
    else:
        path.write_text(rendered, encoding="utf-8")
    counts: dict[str, int] = {}
    for item in inventory["helpers"]:
        status = item["audit_disposition"]
        counts[status] = counts.get(status, 0) + 1
    print(
        json.dumps({"helpers": inventory["helper_count"], "dispositions": counts}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Sphinx configuration for the 5G NR channel-condition documentation site.

The site is built by ``./tools/check docs`` and published by the GitHub Pages
workflow. API pages are generated from the library packages with
``sphinx-autoapi``; Google-style docstrings are read by ``napoleon``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "5G NR Channel Condition"
copyright = "2026, Long Nguyen"
author = "Long Nguyen"
release = "0.2.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_design",
]

root_doc = "index"
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Developer-only material is not part of the user documentation site:
    # research protocols and evidence notes, agent operations, refactor
    # records, and internal audits stay in the repository.
    "research/**",
    "thesis/**",
    "README.md",
    "**/README.md",
    # Frozen research text moved into the guide folders: kept byte-identical to
    # the evidence base, so these pages are not site pages.
    "guide/quickstart.md",
    "guide/luna-execution-guide-2026-11-11.md",
    "architecture/output-tasks-and-channels.md",
    # Repository records kept for auditors, not user pages.
    "operations/REFACTOR_VERIFICATION.md",
    "architecture/NAMING_EXCEPTIONS.md",
    "architecture/module-split-design-notes.md",
]

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
autodoc_typehints = "description"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

autoapi_type = "python"
autoapi_dirs = [
    str(ROOT / package)
    for package in (
        "artifacts",
        "configuration",
        "datasets",
        "propagation",
        "receivers",
        "models",
        "learning",
        "evaluation",
        "experiments",
        "integrations",
        "compute",
        "loslab",
    )
]
autoapi_root = "api/generated"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_python_class_content = "both"
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]

html_theme = "pydata_sphinx_theme"
html_title = project
html_theme_options = {
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "github_url": "https://github.com/nntk072/5g-nr-channel-condition",
}
html_static_path: list[str] = []

REPOSITORY_URL = "https://github.com/nntk072/5g-nr-channel-condition"
_EXCLUDED_DOCUMENT_ROOTS = ("research/", "thesis/")
_MARKDOWN_LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")


def _repository_link(match: re.Match[str], document: Path) -> str:
    """Rewrite a relative link that leaves the published site to a repository URL."""
    target = match.group(2)
    if target.startswith(("http://", "https://", "mailto:", "/")):
        return match.group(0)
    resolved = (document.parent / target).resolve()
    try:
        relative = resolved.relative_to(DOCS_ROOT).as_posix()
    except ValueError:
        relative = resolved.relative_to(ROOT).as_posix()
        url = f"{REPOSITORY_URL}/blob/main/{relative}"
    else:
        if not relative.startswith(_EXCLUDED_DOCUMENT_ROOTS):
            return match.group(0)
        url = f"{REPOSITORY_URL}/blob/main/docs/{relative}"
    return f"{match.group(1)}{url}{match.group(3) or ''}{match.group(4)}"


def _rewrite_repository_links(app, docname, source):
    """Send links to unpublished research notes and repository files to GitHub."""
    document = Path(app.env.doc2path(docname)).resolve()
    source[0] = _MARKDOWN_LINK.sub(lambda m: _repository_link(m, document), source[0])


DOCS_ROOT = Path(__file__).resolve().parent


def setup(app):
    """Register the link rewriting hook with Sphinx."""
    app.connect("source-read", _rewrite_repository_links)
    return {"parallel_read_safe": True}

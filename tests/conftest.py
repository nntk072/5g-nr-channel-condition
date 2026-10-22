"""Initialize Sionna RT before test collection imports the Sionna PHY package."""

from __future__ import annotations

import importlib.metadata


def pytest_configure() -> None:
    """Register Sionna RT's Mitsuba plugins before other tests import Sionna.

    Sionna RT 2.2 registers its custom radio-material plugin for the active
    Mitsuba variant during import. Importing ``sionna`` first (as PHY tests do
    at module scope) and selecting a variant afterward leaves scenes using the
    custom plugin unloadable. The RT native test environment uses LLVM so this
    setup mirrors the portable CPU path used by CI and this development host.
    """
    try:
        importlib.metadata.version("sionna-rt")
    except importlib.metadata.PackageNotFoundError:
        return

    import mitsuba as mi

    variant = "llvm_ad_mono_polarized"
    if variant not in tuple(str(item) for item in mi.variants()):
        return
    if mi.variant() is None:
        mi.set_variant(variant)

    import sionna.rt  # noqa: F401

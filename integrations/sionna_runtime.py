"""Runtime identity helpers for Sionna's split PHY and RT distributions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def sionna_phy_version() -> str:
    """Return the installed PHY distribution version without importing Sionna.

    Sionna publishes the same PHY package namespace as either ``sionna-no-rt``
    or the combined ``sionna`` distribution. Prefer the explicitly PHY-only
    package when both distributions are installed.
    """
    for distribution in ("sionna-no-rt", "sionna"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    raise PackageNotFoundError("Sionna PHY requires sionna-no-rt or sionna")

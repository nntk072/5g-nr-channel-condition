"""Numerical utility tests for the RT convergence runner."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = runpy.run_path(str(ROOT / "scripts" / "run_rt_solver_convergence.py"))
relative_l2_error = RUNNER["relative_l2_error"]


def test_relative_l2_error_uses_reference_norm():
    assert relative_l2_error(
        np.asarray([3 + 0j, 4 + 0j]), np.asarray([0 + 0j, 4 + 0j])
    ) == pytest.approx(0.75)


def test_relative_l2_error_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="aligned complex arrays"):
        relative_l2_error(np.asarray([1 + 0j]), np.asarray([1 + 0j, 2 + 0j]))


def test_relative_l2_error_rejects_zero_reference():
    with pytest.raises(ValueError, match="nonzero energy"):
        relative_l2_error(np.asarray([1 + 0j]), np.asarray([0 + 0j]))

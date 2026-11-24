"""The XGBoost baseline reports its optional extra when the package is absent."""

from __future__ import annotations

import sys

import pytest
from models.classical import make_estimator


def test_xgboost_baseline_names_its_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match="optional 'boosting' dependency extra"):
        make_estimator("xgb")

"""Model-input array domains: dtype families, shape, finiteness and sign rules for each value kind."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import model_input_audit as audit


def _check(value, **kwargs):
    return audit._validated_array(value, name="field", **kwargs)


@pytest.mark.parametrize(
    ("value", "kind", "message"),
    [
        pytest.param(
            np.ones(3, dtype=np.complex64), "real", "must be real-valued", id="complex-as-real"
        ),
        pytest.param(np.ones(3, dtype=bool), "real", "must be real-valued", id="bool-as-real"),
        pytest.param(
            np.ones(3, dtype=np.float64), "complex", "must be complex-valued", id="real-as-complex"
        ),
        pytest.param(
            np.ones(3, dtype=np.int64), "boolean", "must be Boolean-valued", id="int-as-bool"
        ),
        pytest.param(
            np.ones(3, dtype=np.float64), "integer", "must be integer-valued", id="float-as-int"
        ),
        pytest.param(np.ones(3, dtype=bool), "integer", "must be integer-valued", id="bool-as-int"),
    ],
)
def test_value_kind_is_enforced(value, kind, message) -> None:
    with pytest.raises(TypeError, match=message):
        _check(value, value_kind=kind)


def test_matching_value_kinds_are_accepted_and_report_their_shape() -> None:
    assert _check(np.ones((2, 3), dtype=np.complex64), value_kind="complex", ndim=2) == (2, 3)
    assert _check(np.ones(4, dtype=np.float32), value_kind="real") == (4,)
    assert _check(np.ones(4, dtype=bool), value_kind="boolean") == (4,)
    assert _check(np.arange(4, dtype=np.uint8), value_kind="integer") == (4,)


@pytest.mark.parametrize(
    ("value", "kwargs", "error", "message"),
    [
        pytest.param(np.ones((2, 2)), {"ndim": 3}, ValueError, "must have 3 dimensions", id="rank"),
        pytest.param(np.zeros((0, 2)), {}, ValueError, "must have nonempty dimensions", id="empty"),
        pytest.param(
            np.asarray([1, None], dtype=object),
            {},
            TypeError,
            "must be a numeric array",
            id="object",
        ),
        pytest.param(np.asarray(["a"]), {}, TypeError, "must be a numeric array", id="string"),
        pytest.param(
            [1.0, 2.0], {}, TypeError, "must be a NumPy array or dense PyTorch tensor", id="list"
        ),
    ],
)
def test_shape_and_dtype_families_are_refused(value, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        _check(value, value_kind="real", **kwargs)


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, np.inf]),
        np.asarray([1 + 1j, complex(np.nan, 0.0)]),
    ],
    ids=["nan", "inf", "complex-nan"],
)
def test_non_finite_values_are_refused(value) -> None:
    kind = "complex" if np.iscomplexobj(value) else "real"
    with pytest.raises(ValueError, match="only finite values"):
        _check(value, value_kind=kind)


def test_nonnegative_fields_reject_negative_reals_only() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        _check(np.asarray([0.5, -0.1]), value_kind="real", nonnegative=True)
    assert _check(np.asarray([0.0, 2.0]), value_kind="real", nonnegative=True) == (2,)
    # Complex and Boolean fields are not sign-checked: the sign is not defined for them.
    assert _check(np.asarray([-1.0 + 0j]), value_kind="complex", nonnegative=True) == (1,)


def test_field_roster_must_come_from_a_dataclass() -> None:
    with pytest.raises(TypeError, match="audited dataclass contract"):
        audit._field_names({"not": "a dataclass"})


def test_real_scalars_are_finite_and_respect_sign_constraints() -> None:
    audit._validated_real_scalar(0.25, name="snr_db")
    for value, kwargs, message in [
        (True, {}, "must be a real numeric scalar"),
        ("0.1", {}, "must be a real numeric scalar"),
        (float("nan"), {}, "must be finite"),
        (-1.0, {"nonnegative": True}, "must be nonnegative"),
        (0.0, {"positive": True}, "must be positive"),
    ]:
        with pytest.raises((TypeError, ValueError), match=message):
            audit._validated_real_scalar(value, name="x", **kwargs)


def test_torch_tensors_follow_the_same_domains_when_torch_is_installed() -> None:
    torch = pytest.importorskip("torch")

    assert _check(torch.ones(2, 3, dtype=torch.complex64), value_kind="complex", ndim=2) == (2, 3)
    assert _check(torch.ones(4), value_kind="real") == (4,)
    assert _check(torch.ones(4, dtype=torch.bool), value_kind="boolean") == (4,)
    assert _check(torch.arange(4), value_kind="integer") == (4,)
    with pytest.raises(TypeError, match="must be real-valued"):
        _check(torch.ones(4, dtype=torch.complex64), value_kind="real")
    with pytest.raises(TypeError, match="must be integer-valued"):
        _check(torch.ones(4), value_kind="integer")
    with pytest.raises(ValueError, match="must be nonnegative"):
        _check(torch.tensor([0.5, -1.0]), value_kind="real", nonnegative=True)
    with pytest.raises(ValueError, match="only finite values"):
        _check(torch.tensor([1.0, float("inf")]), value_kind="real")
    with pytest.raises(ValueError, match="must have 2 dimensions"):
        _check(torch.ones(4), value_kind="real", ndim=2)
    with pytest.raises(ValueError, match="nonempty dimensions"):
        _check(torch.ones(0, 2), value_kind="real")
    with pytest.raises(TypeError, match="dense numeric array"):
        _check(torch.ones(2, 2).to_sparse(), value_kind="real")

from types import SimpleNamespace

import integrations.sionna.network_phy.records as records
import pytest
import torch


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([True], "not Boolean"),
        (torch.tensor([1], dtype=torch.bool), "not Boolean or complex"),
        (torch.tensor([1 + 1j]), "not Boolean or complex"),
        (torch.sparse_coo_tensor([[0]], [1.0], (1,)), "dense real-valued vector"),
        (torch.tensor([1], dtype=torch.uint16), "real numeric values"),
        (["not a number"], "real numeric values"),
        ([float("inf")], "finite and nonnegative"),
        ([-1.0], "finite and nonnegative"),
        ([1.0, 2.0], "one variance per receiver"),
    ],
)
def test_nonnegative_vector_validator_rejects_invalid_real_data(value, message):
    with pytest.raises(ValueError, match=message):
        records._validated_nonnegative_vector(
            torch,
            value,
            size=1,
            device=torch.device("cpu"),
            name="noise_variance_w_per_re",
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ((True,), "not Boolean"),
        (torch.tensor([1], dtype=torch.bool), "not Boolean or complex"),
        (torch.tensor([1 + 1j]), "not Boolean or complex"),
        (torch.sparse_coo_tensor([[0]], [1.0], (1,)), "dense real-valued array"),
        (torch.tensor([1], dtype=torch.uint16), "real numeric values"),
        (["not a number"], "real numeric values"),
        ([1.0], r"shape \(2,\)"),
        ([float("nan"), 0.0], "must be finite"),
        ([-1.0, 0.0], "must be nonnegative"),
    ],
)
def test_real_value_validator_rejects_invalid_arrays(value, message):
    with pytest.raises(ValueError, match=message):
        records._validated_real_values(
            torch,
            value,
            shape=(2,),
            device=torch.device("cpu"),
            name="path_loss_db",
            nonnegative=True,
        )


def test_real_value_validators_convert_after_dtype_probe_and_report_conversion_errors(monkeypatch):
    original_as_tensor = torch.as_tensor
    calls = 0

    def fail_conversion(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected conversion failure")
        return original_as_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", fail_conversion)
    with pytest.raises(ValueError, match="real numeric values"):
        records._validated_nonnegative_vector(
            torch,
            [1.0],
            size=1,
            device=torch.device("cpu"),
            name="transmit_power_w_per_re",
        )

    calls = 0
    with pytest.raises(ValueError, match="real numeric values"):
        records._validated_real_values(
            torch,
            [1.0],
            shape=(1,),
            device=torch.device("cpu"),
            name="gain_db",
        )


def _composition():
    contributions = torch.ones((1, 1, 1, 2, 1, 2), dtype=torch.complex64)
    received = contributions.sum(dim=3)
    return SimpleNamespace(
        source_contributions=contributions,
        received_grid=received,
        transmit_power_w_per_re=torch.ones(2),
    )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("source_contributions", None, TypeError, "complex PyTorch source contributions"),
        ("source_contributions", torch.ones((1, 1)), TypeError, "complex PyTorch"),
        (
            "source_contributions",
            torch.ones((1, 1), dtype=torch.complex64),
            ValueError,
            "source contributions must have",
        ),
        ("received_grid", None, ValueError, "received_grid must be complex"),
        ("received_grid", torch.ones((1, 1)), ValueError, "received_grid must be complex"),
        (
            "received_grid",
            torch.ones((1, 1, 1, 1), dtype=torch.complex64),
            ValueError,
            "received_grid must be complex",
        ),
        (
            "received_grid",
            torch.zeros((1, 1, 1, 2, 2), dtype=torch.complex64),
            ValueError,
            "shape does not match",
        ),
        (
            "received_grid",
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex128),
            ValueError,
            "share device and dtype",
        ),
        ("transmit_power_w_per_re", None, ValueError, "one value per transmitter"),
        ("transmit_power_w_per_re", (1.0, 1.0), ValueError, "one value per transmitter"),
        ("transmit_power_w_per_re", torch.ones(1), ValueError, "one value per transmitter"),
        ("transmit_power_w_per_re", torch.ones(2, dtype=torch.int64), ValueError, "real tensor"),
        (
            "transmit_power_w_per_re",
            torch.tensor([1.0, -1.0]),
            ValueError,
            "finite and nonnegative",
        ),
    ],
)
def test_composition_validator_rejects_inconsistent_records(field, value, error, message):
    composition = _composition()
    setattr(composition, field, value)
    with pytest.raises(error, match=message):
        records._validate_composition(torch, composition)


def test_composition_validator_rejects_nonfinite_values():
    composition = _composition()
    composition.received_grid[0, 0, 0, 0, 0] = complex(float("inf"), 0)
    with pytest.raises(ValueError, match="must be finite"):
        records._validate_composition(torch, composition)

"""Dependency-isolated input contracts for the NR PUSCH receiver adapter."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "Boolean or complex"),
        (1.0 + 0.5j, "Boolean or complex"),
        (-0.1, "finite and nonnegative"),
        (float("nan"), "finite and nonnegative"),
        ("noise", "real numeric values"),
    ],
)
def test_pusch_noise_variance_rejects_nonphysical_values(value: object, message: str) -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    with pytest.raises(ValueError, match=message):
        _validated_noise_variance(torch, value, device="cpu")


def test_pusch_noise_variance_rejects_float32_overflow() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    value = torch.tensor(1e100, dtype=torch.float64)
    with pytest.raises(ValueError, match="representable as finite float32"):
        _validated_noise_variance(torch, value, device="cpu")


def test_pusch_noise_variance_rejects_positive_float32_underflow() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    value = torch.tensor(1e-100, dtype=torch.float64)
    with pytest.raises(ValueError, match="underflow to zero"):
        _validated_noise_variance(torch, value, device="cpu")


def test_pusch_noise_variance_rejects_python_scalar_underflow() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    with pytest.raises(ValueError, match="underflow to zero"):
        _validated_noise_variance(torch, 1e-100, device="cpu")


def test_pusch_noise_variance_accepts_nonnegative_real_tensors() -> None:
    torch = pytest.importorskip("torch")
    from integrations.sionna.nr_pusch import _validated_noise_variance

    value = torch.tensor([0, 2], dtype=torch.int32)
    result = _validated_noise_variance(torch, value, device="cpu")

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0]))

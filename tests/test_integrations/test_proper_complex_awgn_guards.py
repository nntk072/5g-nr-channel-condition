"""Receiver noise generation: torch availability, reference type, and noise-variance domain checks."""

from __future__ import annotations

import sys

import pytest
from integrations.sionna.noise import proper_complex_awgn

torch = pytest.importorskip("torch")


def _generator():
    return torch.Generator().manual_seed(7)


def test_noise_requires_torch_when_it_is_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="Install PyTorch to generate receiver noise"):
        proper_complex_awgn(None, 0.1, generator=None)


def test_reference_must_be_a_complex_tensor() -> None:
    with pytest.raises(TypeError, match="complex PyTorch tensor"):
        proper_complex_awgn(torch.ones(2), 0.1, generator=_generator())


@pytest.mark.parametrize(
    ("variance", "message"),
    [
        pytest.param("not-a-number", "must contain real numeric values", id="string"),
        pytest.param(torch.ones(2, dtype=torch.complex64), "not Boolean or complex", id="complex"),
        pytest.param(torch.tensor([True, False]), "not Boolean or complex", id="bool"),
        pytest.param(-1.0, "finite and nonnegative", id="negative"),
        pytest.param(float("nan"), "finite and nonnegative", id="nan"),
    ],
)
def test_noise_variance_domain_is_enforced(variance, message) -> None:
    reference = torch.zeros(4, dtype=torch.complex64)
    with pytest.raises((TypeError, ValueError), match=message):
        proper_complex_awgn(reference, variance, generator=_generator())


def test_noise_has_the_requested_mean_power() -> None:
    reference = torch.zeros(20000, dtype=torch.complex64)
    noise = proper_complex_awgn(reference, 0.5, generator=_generator())
    assert noise.dtype == torch.complex64
    assert abs(float(torch.mean(torch.abs(noise) ** 2)) - 0.5) < 0.02

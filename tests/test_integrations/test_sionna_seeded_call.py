"""Seeded Sionna calls: the RNG state is set for the call and restored afterwards."""

from __future__ import annotations

import sys

import pytest
from integrations.sionna.random import call_with_sionna_seed


def test_missing_sionna_phy_is_reported_as_a_runtime_requirement(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sionna.phy", None)
    with pytest.raises(RuntimeError, match="Sionna PHY is required"):
        call_with_sionna_seed("cpu", 1, lambda: None)


@pytest.mark.parametrize("seed", [True, -1, 1.5, "7"])
def test_seed_must_be_a_nonnegative_integer(seed) -> None:
    pytest.importorskip("sionna.phy")
    with pytest.raises(ValueError, match="nonnegative integer"):
        call_with_sionna_seed("cpu", seed, lambda: None)


def test_call_runs_with_the_seed_and_restores_the_generator_state() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from sionna.phy import config

    generator = config.torch_rng("cpu")
    generator.manual_seed(999)
    before = generator.get_state().clone()

    def draw():
        return torch.rand(3, generator=generator)

    first = call_with_sionna_seed("cpu", 42, draw)
    again = call_with_sionna_seed("cpu", 42, draw)
    other = call_with_sionna_seed("cpu", 43, draw)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert torch.equal(generator.get_state(), before)


def test_state_is_restored_even_when_the_call_fails() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from sionna.phy import config

    generator = config.torch_rng("cpu")
    generator.manual_seed(5)
    before = generator.get_state().clone()

    def failing():
        torch.rand(2, generator=generator)
        raise ValueError("solver failed")

    with pytest.raises(ValueError, match="solver failed"):
        call_with_sionna_seed("cpu", 8, failing)
    assert torch.equal(generator.get_state(), before)

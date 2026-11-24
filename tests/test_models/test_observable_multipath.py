"""Two-path observation model: rendering, likelihoods, CRLB floor and decoder checkpoints."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.observable_multipath import (  # noqa: E402
    build_two_path_uncertainty_decoder,
    complex_pilot_gaussian_nll,
    heteroscedastic_delay_nll,
    identifiability_scale_floor_loss,
    load_two_path_uncertainty_decoder,
    relative_delay_crlb_scale_floor,
    render_two_path_cfr,
)

FREQUENCIES = torch.linspace(-2e6, 2e6, 16, dtype=torch.float64)


def _gains(batch: int = 2) -> torch.Tensor:
    return torch.complex(
        torch.linspace(0.8, 1.2, batch * 2, dtype=torch.float64).reshape(batch, 2),
        torch.linspace(-0.3, 0.4, batch * 2, dtype=torch.float64).reshape(batch, 2),
    )


def test_render_matches_the_two_path_formula_and_fixes_the_reference_path() -> None:
    gains = _gains()
    delay = torch.tensor([50e-9, 120e-9], dtype=torch.float64)
    cfr = render_two_path_cfr(FREQUENCIES, gains, delay)
    expected = gains[:, :1] + gains[:, 1:2] * torch.exp(
        -2j * math.pi * FREQUENCIES[None, :] * delay[:, None]
    )
    assert torch.allclose(cfr, expected)
    # Zero relative delay collapses both paths to one coherent sum: the gauge is fixed here.
    zero = render_two_path_cfr(FREQUENCIES, gains, torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(zero, gains.sum(dim=1, keepdim=True).expand_as(zero))


def test_render_accepts_batched_frequencies_and_column_delays() -> None:
    gains = _gains()
    batched = FREQUENCIES.expand(2, -1).contiguous()
    delay = torch.tensor([[10e-9], [20e-9]], dtype=torch.float64)
    shared = render_two_path_cfr(FREQUENCIES, gains, delay)
    assert torch.allclose(render_two_path_cfr(batched, gains, delay), shared)


@pytest.mark.parametrize(
    ("frequencies", "gains", "delay", "error"),
    [
        (FREQUENCIES.numpy(), _gains(), torch.zeros(2), TypeError),
        (FREQUENCIES, _gains().real, torch.zeros(2), ValueError),
        (FREQUENCIES[:1], _gains(), torch.zeros(2), ValueError),
        (FREQUENCIES, _gains(), torch.zeros(3), ValueError),
        (FREQUENCIES.expand(3, -1), _gains(), torch.zeros(2), ValueError),
        (FREQUENCIES, _gains(), torch.tensor([np.nan, 0.0]), ValueError),
    ],
)
def test_render_rejects_invalid_inputs(frequencies, gains, delay, error) -> None:
    with pytest.raises(error):
        render_two_path_cfr(frequencies, gains, delay)


def test_complex_nll_has_the_gaussian_value_and_ignores_unobserved_tones() -> None:
    observed = torch.complex(
        torch.zeros(1, 4, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64)
    )
    predicted = observed.clone()
    mask = torch.tensor([[True, True, False, False]])
    variance = torch.tensor([2.0], dtype=torch.float64)
    nll = complex_pilot_gaussian_nll(observed, predicted, variance, mask)
    assert nll.item() == pytest.approx(math.log(math.pi * 2.0))
    # Corrupting an unobserved tone must not change the likelihood.
    corrupted = predicted.clone()
    corrupted[0, 3] = 100.0
    assert complex_pilot_gaussian_nll(observed, corrupted, variance, mask).item() == pytest.approx(
        nll.item()
    )


def test_complex_nll_penalizes_residuals_and_accepts_per_tone_variance() -> None:
    observed = torch.ones(2, 3, dtype=torch.complex128)
    predicted = observed + 1.0
    mask = torch.ones(2, 3, dtype=torch.bool)
    scalar = complex_pilot_gaussian_nll(
        observed, predicted, torch.tensor([1.0, 1.0], dtype=torch.float64), mask
    )
    per_tone = complex_pilot_gaussian_nll(
        observed, predicted, torch.ones(2, 3, dtype=torch.float64), mask
    )
    assert scalar.item() == pytest.approx(1.0 + math.log(math.pi))
    assert per_tone.item() == pytest.approx(scalar.item())


@pytest.mark.parametrize(
    ("observed", "predicted", "mask", "variance", "message"),
    [
        (
            torch.ones(2, 3, dtype=torch.complex128),
            torch.ones(2, 2, dtype=torch.complex128),
            torch.ones(2, 3, dtype=torch.bool),
            1.0,
            "matching complex",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
            1.0,
            "matching complex",
        ),
        (
            torch.full((1, 2), complex(np.nan, 0.0), dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.bool),
            1.0,
            "finite",
        ),
        (
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2),
            1.0,
            "boolean",
        ),
        (
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.complex128),
            torch.zeros(1, 2, dtype=torch.bool),
            1.0,
            "at least one observed",
        ),
        (
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, dtype=torch.float64),
            "finite and positive",
        ),
        (
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.complex128),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(3),
            "batch] or",
        ),
    ],
)
def test_complex_nll_rejects_invalid_inputs(observed, predicted, mask, variance, message) -> None:
    with pytest.raises(ValueError, match=message):
        complex_pilot_gaussian_nll(observed, predicted, variance, mask)


def test_heteroscedastic_nll_is_minimal_at_the_predicted_delay() -> None:
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    at_truth = heteroscedastic_delay_nll(target, target, torch.zeros(2, dtype=torch.float64))
    assert at_truth.item() == pytest.approx(0.0)
    off = heteroscedastic_delay_nll(target, target + 1.0, torch.zeros(2, dtype=torch.float64))
    assert off.item() == pytest.approx(0.5)
    # Integer targets are promoted to floating point before the likelihood.
    assert heteroscedastic_delay_nll(
        torch.tensor([1, 2]), torch.tensor([1.0, 2.0]), torch.zeros(2)
    ).item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("target", "mean", "log_scale", "message"),
    [
        (torch.ones(2), torch.ones(3), torch.zeros(2), "identical shapes"),
        (torch.empty(0), torch.empty(0), torch.empty(0), "nonempty and finite"),
        (torch.tensor([np.nan, 1.0]), torch.ones(2), torch.zeros(2), "nonempty and finite"),
        (torch.ones(2), torch.tensor([np.inf, 1.0]), torch.zeros(2), "must be finite"),
        (torch.ones(2), torch.ones(2), torch.tensor([np.nan, 0.0]), "must be finite"),
    ],
)
def test_heteroscedastic_nll_rejects_invalid_inputs(target, mean, log_scale, message) -> None:
    with pytest.raises(ValueError, match=message):
        heteroscedastic_delay_nll(target, mean, log_scale)


def test_heteroscedastic_nll_clamps_extreme_log_scales() -> None:
    value = heteroscedastic_delay_nll(
        torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1000.0])
    )
    assert math.isfinite(value.item())
    assert value.item() == pytest.approx(
        0.5 * math.exp(-30.0 * 2) * 0 + 30.0 + 0.5 * (1.0 * math.exp(-30.0)) ** 2, rel=1e-6
    )


def _crlb_inputs(batch: int = 2, tones: int = 12):
    gains = _gains(batch)
    delay = torch.tensor([60e-9, 90e-9], dtype=torch.float64)[:batch]
    mask = torch.ones(batch, tones, dtype=torch.bool)
    frequencies = torch.linspace(-3e6, 3e6, tones, dtype=torch.float64)
    variance = torch.full((batch,), 1e-3, dtype=torch.float64)
    return gains, delay, frequencies, variance, mask


def test_crlb_floor_is_positive_and_capped_by_the_declared_delay_support() -> None:
    gains, delay, frequencies, variance, mask = _crlb_inputs()
    floor = relative_delay_crlb_scale_floor(gains, delay, frequencies, variance, mask, 5e-6)
    assert floor.shape == (2, 1)
    assert torch.all(floor > 0) and torch.all(floor <= 5e-6)
    # Less noise means more information, so the floor must shrink.
    quieter = relative_delay_crlb_scale_floor(
        gains, delay, frequencies, variance / 100.0, mask, 5e-6
    )
    assert torch.all(quieter < floor)
    # A tiny support caps the floor exactly.
    capped = relative_delay_crlb_scale_floor(gains, delay, frequencies, variance, mask, 1e-12)
    assert torch.allclose(capped, torch.full_like(capped, 1e-12))


def test_crlb_floor_accepts_batched_frequencies_and_column_delays() -> None:
    gains, delay, frequencies, variance, mask = _crlb_inputs()
    shared = relative_delay_crlb_scale_floor(gains, delay, frequencies, variance, mask, 5e-6)
    batched = relative_delay_crlb_scale_floor(
        gains, delay[:, None], frequencies.expand(2, -1), variance, mask, 5e-6
    )
    assert torch.allclose(batched, shared)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gains": _gains().real}, "complex tensor"),
        ({"mask": torch.ones(3, 12, dtype=torch.bool)}, "match the gain batch"),
        ({"mask": torch.ones(2, 12)}, "boolean with at least two"),
        (
            {
                "mask": torch.cat(
                    [torch.zeros(1, 12, dtype=torch.bool), torch.ones(1, 12, dtype=torch.bool)]
                )
            },
            "boolean with at least two",
        ),
        ({"frequencies": torch.ones(2, 5, dtype=torch.float64)}, "match the pilot mask"),
        ({"delay": torch.zeros(3, dtype=torch.float64)}, "shape \\[batch\\] or"),
        ({"variance": torch.zeros(2, dtype=torch.float64)}, "finite and positive"),
        ({"max_delay": 0.0}, "finite and positive"),
        ({"frequencies": torch.full((12,), np.nan, dtype=torch.float64)}, "must be finite"),
    ],
)
def test_crlb_floor_rejects_invalid_inputs(kwargs, message) -> None:
    gains, delay, frequencies, variance, mask = _crlb_inputs()
    args = {
        "gains": gains,
        "delay": delay,
        "frequencies": frequencies,
        "variance": variance,
        "mask": mask,
        "max_delay": 5e-6,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        relative_delay_crlb_scale_floor(
            args["gains"],
            args["delay"],
            args["frequencies"],
            args["variance"],
            args["mask"],
            args["max_delay"],
        )


def test_identifiability_loss_penalizes_only_scales_below_the_floor() -> None:
    gains, delay, frequencies, variance, mask = _crlb_inputs()
    floor = relative_delay_crlb_scale_floor(gains, delay, frequencies, variance, mask, 5e-6)
    args = (gains, delay, frequencies, variance, mask, 5e-6)
    understated = identifiability_scale_floor_loss(floor / 10.0, *args)
    overstated = identifiability_scale_floor_loss(floor * 10.0, *args)
    assert understated.item() > 0.0
    assert overstated.item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        identifiability_scale_floor_loss(torch.zeros(2, 1, dtype=torch.float64), *args)


def _decoder_inputs(batch: int = 3, tones: int = 8):
    cfr = torch.complex(
        torch.linspace(0.1, 1.0, batch * tones, dtype=torch.float32).reshape(batch, tones),
        torch.linspace(-0.5, 0.5, batch * tones, dtype=torch.float32).reshape(batch, tones),
    )
    mask = torch.ones(batch, tones, dtype=torch.bool)
    frequencies = torch.linspace(0.0, 4e6, tones, dtype=torch.float32)
    variance = torch.full((batch,), 0.5, dtype=torch.float32)
    return cfr, mask, frequencies, variance


def test_decoder_outputs_bounded_delay_and_positive_scale() -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(16,))
    cfr, mask, frequencies, variance = _decoder_inputs()
    out = decoder(cfr, mask, frequencies, variance)
    assert out["gains"].shape == (3, 2) and out["gains"].is_complex()
    assert torch.all(out["relative_delay_s"] >= 0) and torch.all(out["relative_delay_s"] <= 2e-6)
    assert torch.all(out["relative_delay_std_s"] > 0)
    assert torch.allclose(out["relative_delay_s"], out["delay_fraction"] * 2e-6)


def test_decoder_accepts_batched_frequencies() -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    cfr, mask, frequencies, variance = _decoder_inputs()
    shared = decoder(cfr, mask, frequencies, variance)
    batched = decoder(cfr, mask, frequencies.expand(3, -1), variance)
    assert torch.allclose(shared["relative_delay_s"], batched["relative_delay_s"], atol=1e-6)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cfr": torch.ones(3, 7, dtype=torch.complex64)}, "shape \\[batch, n_tones\\]"),
        ({"cfr": torch.ones(3, 8)}, "must be complex"),
        ({"mask": torch.ones(3, 8)}, "boolean and match"),
        (
            {
                "mask": torch.cat(
                    [torch.zeros(1, 8, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool)]
                )
            },
            "at least two pilot",
        ),
        ({"freq": torch.ones(5)}, "pilot-tone shape"),
        ({"freq": torch.full((8,), np.nan)}, "pilot-tone shape"),
        ({"freq": torch.zeros(8)}, "nonzero bandwidth"),
        ({"variance": torch.ones(2)}, "shape \\[batch\\]"),
        ({"variance": torch.zeros(3)}, "finite and positive"),
    ],
)
def test_decoder_rejects_invalid_inputs(change, message) -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    cfr, mask, frequencies, variance = _decoder_inputs()
    args = {"cfr": cfr, "mask": mask, "freq": frequencies, "variance": variance}
    args.update(change)
    with pytest.raises(ValueError, match=message):
        decoder(args["cfr"], args["mask"], args["freq"], args["variance"])


def test_decoder_rejects_nonfinite_observations() -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    cfr, mask, frequencies, variance = _decoder_inputs()
    bad = cfr.clone()
    bad[0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite values"):
        decoder(bad, mask, frequencies, variance)


@pytest.mark.parametrize(
    ("n_tones", "max_delay", "hidden", "message"),
    [
        (True, 2e-6, (8,), "integer >= 2"),
        (1, 2e-6, (8,), "integer >= 2"),
        (8, 0.0, (8,), "finite and positive"),
        (8, 2e-6, (), "positive integers"),
        (8, 2e-6, (0,), "positive integers"),
        (8, 2e-6, (True,), "positive integers"),
    ],
)
def test_decoder_build_rejects_invalid_architecture(n_tones, max_delay, hidden, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_two_path_uncertainty_decoder(n_tones, max_delay, hidden_dims=hidden)


def _save_state(decoder, path):
    np.savez(
        path,
        **{name: tensor.detach().cpu().numpy() for name, tensor in decoder.state_dict().items()},
    )


def test_checkpoint_round_trip_restores_outputs_and_marks_contract_status(tmp_path) -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    checkpoint = tmp_path / "decoder.npz"
    _save_state(decoder, checkpoint)
    loaded = load_two_path_uncertainty_decoder(checkpoint, 8, 2e-6, hidden_dims=(8,))
    cfr, mask, frequencies, variance = _decoder_inputs()
    assert not loaded.training
    assert loaded.artifact_contract_status == "external_input_contract_required"
    assert torch.allclose(
        decoder.eval()(cfr, mask, frequencies, variance)["relative_delay_s"],
        loaded(cfr, mask, frequencies, variance)["relative_delay_s"],
    )


def test_checkpoint_loader_requires_a_contract_when_asked(tmp_path) -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    checkpoint = tmp_path / "decoder.npz"
    _save_state(decoder, checkpoint)
    with pytest.raises(ValueError, match="input contract is required"):
        load_two_path_uncertainty_decoder(
            checkpoint, 8, 2e-6, hidden_dims=(8,), require_contract=True
        )
    with pytest.raises(ValueError, match="input contract is required"):
        load_two_path_uncertainty_decoder(checkpoint, 8, 2e-6, hidden_dims=(8,), dataset=object())


def test_checkpoint_loader_rejects_missing_and_mismatched_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        load_two_path_uncertainty_decoder(tmp_path / "absent.npz", 8, 2e-6, hidden_dims=(8,))
    not_npz = tmp_path / "garbage.npz"
    not_npz.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="invalid decoder checkpoint"):
        load_two_path_uncertainty_decoder(not_npz, 8, 2e-6, hidden_dims=(8,))

    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    state = {name: tensor.detach().cpu().numpy() for name, tensor in decoder.state_dict().items()}
    renamed = tmp_path / "renamed.npz"
    np.savez(renamed, **{"extra." + name: value for name, value in state.items()})
    with pytest.raises(ValueError, match="parameter names do not match"):
        load_two_path_uncertainty_decoder(renamed, 8, 2e-6, hidden_dims=(8,))

    name = next(iter(state))
    wrong_shape = dict(state)
    wrong_shape[name] = np.zeros((1,), dtype=state[name].dtype)
    shaped = tmp_path / "shaped.npz"
    np.savez(shaped, **wrong_shape)
    with pytest.raises(ValueError, match="incompatible"):
        load_two_path_uncertainty_decoder(shaped, 8, 2e-6, hidden_dims=(8,))

    nonfinite = dict(state)
    nonfinite[name] = np.full(state[name].shape, np.nan, dtype=state[name].dtype)
    bad = tmp_path / "nonfinite.npz"
    np.savez(bad, **nonfinite)
    with pytest.raises(ValueError, match="nonfinite"):
        load_two_path_uncertainty_decoder(bad, 8, 2e-6, hidden_dims=(8,))


def test_checkpoint_loader_rejects_a_symlinked_checkpoint(tmp_path) -> None:
    decoder = build_two_path_uncertainty_decoder(8, 2e-6, hidden_dims=(8,))
    real = tmp_path / "real.npz"
    _save_state(decoder, real)
    link = tmp_path / "link.npz"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="regular file"):
        load_two_path_uncertainty_decoder(link, 8, 2e-6, hidden_dims=(8,))

"""Differentiable two-path CFR rendering and uncertainty decoder contracts."""

from __future__ import annotations

import pytest
from models.observable_multipath import (
    build_two_path_uncertainty_decoder,
    complex_pilot_gaussian_nll,
    heteroscedastic_delay_nll,
    identifiability_scale_floor_loss,
    load_two_path_uncertainty_decoder,
    relative_delay_crlb_scale_floor,
    render_two_path_cfr,
)

torch = pytest.importorskip("torch")


def test_two_path_renderer_matches_closed_form_and_backpropagates():
    frequencies = torch.tensor([-2e6, -1e6, 0.0, 1e6, 2e6], dtype=torch.float64)
    gains = torch.tensor([[1.0 + 0.2j, 0.4 - 0.3j]], dtype=torch.complex128, requires_grad=True)
    delay = torch.tensor([0.2e-6], dtype=torch.float64, requires_grad=True)

    result = render_two_path_cfr(frequencies, gains, delay)
    expected = gains[:, :1] + gains[:, 1:2] * torch.exp(
        -2j * torch.pi * frequencies[None, :] * delay[:, None]
    )
    torch.testing.assert_close(result, expected)
    result.abs().square().sum().backward()
    assert gains.grad is not None and torch.isfinite(gains.grad).all()
    assert delay.grad is not None and torch.isfinite(delay.grad).all()


def test_decoder_returns_bounded_delay_and_positive_uncertainty_with_gradients():
    torch.manual_seed(12)
    frequencies = torch.linspace(-3e6, 3e6, 8)
    gains = torch.tensor([[1.0 + 0.1j, 0.35 - 0.2j], [0.6 - 0.3j, 0.2 + 0.5j]])
    truth_delay = torch.tensor([0.18e-6, 0.42e-6])
    truth = render_two_path_cfr(frequencies, gains, truth_delay)
    variance = torch.full((2,), 1e-3)
    mask = torch.ones((2, 8), dtype=torch.bool)
    model = build_two_path_uncertainty_decoder(8, 1e-6, hidden_dims=(24, 12))

    output = model(truth, mask, frequencies, variance)
    reconstructed = render_two_path_cfr(frequencies, output["gains"], output["relative_delay_s"])
    objective = complex_pilot_gaussian_nll(truth, reconstructed, variance, mask)
    objective = objective + heteroscedastic_delay_nll(
        truth_delay,
        output["relative_delay_s"],
        torch.log(output["relative_delay_std_s"]),
    )
    objective.backward()

    assert torch.all(output["relative_delay_s"] > 0)
    assert torch.all(output["relative_delay_s"] < 1e-6)
    assert torch.all(output["relative_delay_std_s"] > 0)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pilot_likelihood_ignores_unobserved_tones():
    observed = torch.tensor([[1 + 0j, 2 + 1j, 3 - 1j]], dtype=torch.complex64)
    predicted = torch.zeros_like(observed)
    mask = torch.tensor([[True, False, True]])
    altered = observed.clone()
    altered[0, 1] = 1e6 - 1e6j

    first = complex_pilot_gaussian_nll(observed, predicted, torch.ones(1), mask)
    second = complex_pilot_gaussian_nll(altered, predicted, torch.ones(1), mask)

    torch.testing.assert_close(first, second)


def test_local_delay_crlb_tracks_bandwidth_and_noise_and_is_differentiable():
    gains = torch.tensor([[1.0 + 0.2j, 0.5 - 0.1j]], dtype=torch.complex128)
    delay = torch.tensor([0.2e-6], dtype=torch.float64)
    mask = torch.ones((1, 32), dtype=torch.bool)
    narrow_frequencies = torch.linspace(-1e6, 1e6, 32, dtype=torch.float64)
    wide_frequencies = torch.linspace(-8e6, 8e6, 32, dtype=torch.float64)
    low_noise = torch.tensor([1e-4], dtype=torch.float64)
    high_noise = torch.tensor([1e-2], dtype=torch.float64)

    narrow = relative_delay_crlb_scale_floor(
        gains, delay, narrow_frequencies, low_noise, mask, 1e-6
    )
    wide = relative_delay_crlb_scale_floor(gains, delay, wide_frequencies, low_noise, mask, 1e-6)
    noisy = relative_delay_crlb_scale_floor(gains, delay, wide_frequencies, high_noise, mask, 1e-6)
    assert torch.all(wide < narrow)
    assert torch.all(noisy > wide)

    predicted_scale = torch.full((1,), 1e-15, dtype=torch.float64, requires_grad=True)
    loss = identifiability_scale_floor_loss(
        predicted_scale,
        gains,
        delay,
        wide_frequencies,
        low_noise,
        mask,
        1e-6,
    )
    loss.backward()
    assert loss > 0
    assert predicted_scale.grad is not None and torch.isfinite(predicted_scale.grad).all()


def test_decoder_rejects_too_sparse_or_nonfinite_inputs():
    model = build_two_path_uncertainty_decoder(3, 1e-6, hidden_dims=(8,))
    with pytest.raises(ValueError, match="at least two"):
        model(
            torch.ones((1, 3), dtype=torch.complex64),
            torch.tensor([[True, False, False]]),
            torch.arange(3, dtype=torch.float32),
            torch.ones(1),
        )


def test_npz_checkpoint_round_trip_is_pickle_free_and_architecture_checked(tmp_path):
    model = build_two_path_uncertainty_decoder(4, 1e-6, hidden_dims=(8,))
    path = tmp_path / "state.npz"
    import numpy as np

    np.savez_compressed(
        path,
        **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
    )
    loaded = load_two_path_uncertainty_decoder(path, 4, 1e-6, hidden_dims=(8,))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)

    with pytest.raises(ValueError, match="architecture"):
        load_two_path_uncertainty_decoder(path, 4, 1e-6, hidden_dims=(9,))
    with pytest.raises(ValueError, match="finite"):
        render_two_path_cfr(
            torch.tensor([0.0, 1.0]),
            torch.tensor([[1 + 0j, 1 + 0j]]),
            torch.tensor([float("nan")]),
        )


def test_decoder_forward_and_backward_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda")
    frequencies = torch.linspace(-2e6, 2e6, 6, device=device)
    observed = torch.ones((2, 6), dtype=torch.complex64, device=device)
    mask = torch.ones((2, 6), dtype=torch.bool, device=device)
    variance = torch.full((2,), 0.01, device=device)
    model = build_two_path_uncertainty_decoder(6, 1e-6, hidden_dims=(12,)).to(device)

    output = model(observed, mask, frequencies, variance)
    rendered = render_two_path_cfr(frequencies, output["gains"], output["relative_delay_s"])
    loss = complex_pilot_gaussian_nll(observed, rendered, variance, mask)
    loss = loss + heteroscedastic_delay_nll(
        torch.full((2,), 0.2e-6, device=device),
        output["relative_delay_s"],
        torch.log(output["relative_delay_std_s"]),
    )
    loss.backward()

    assert output["gains"].is_cuda
    assert all(parameter.grad is not None for parameter in model.parameters())

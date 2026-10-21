import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.tdl_pusch import seeded_complex_awgn  # noqa: E402


def test_seeded_awgn_preserves_common_prefix_when_sample_length_changes():
    short = seeded_complex_awgn(
        (2, 13),
        device="cpu",
        noise_variance=0.25,
        real_seed=100,
        imag_seed=101,
    )
    long = seeded_complex_awgn(
        (2, 17),
        device="cpu",
        noise_variance=0.25,
        real_seed=100,
        imag_seed=101,
    )
    torch.testing.assert_close(short, long[..., :13], rtol=0, atol=0)


@pytest.mark.native_sionna
def test_seeded_awgn_common_prefix_transfers_to_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    short = seeded_complex_awgn(
        (2, 13),
        device="cuda:0",
        noise_variance=0.25,
        real_seed=100,
        imag_seed=101,
    )
    long = seeded_complex_awgn(
        (2, 17),
        device="cuda:0",
        noise_variance=0.25,
        real_seed=100,
        imag_seed=101,
    )
    torch.testing.assert_close(short, long[..., :13], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": ()},
        {"shape": (2, 0)},
        {"shape": (2, True)},
        {"noise_variance": -1.0},
        {"noise_variance": float("nan")},
        {"real_seed": -1},
        {"imag_seed": True},
    ],
)
def test_seeded_awgn_rejects_invalid_inputs(kwargs):
    values = {
        "shape": (2, 8),
        "device": "cpu",
        "noise_variance": 0.5,
        "real_seed": 1,
        "imag_seed": 2,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        seeded_complex_awgn(**values)

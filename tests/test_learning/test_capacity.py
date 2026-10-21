from __future__ import annotations

import pytest
from models.capacity import capacity_spec


def test_capacity_presets_are_ordered_and_reject_unknown_names() -> None:
    small, medium, large = (capacity_spec(name) for name in ("small", "medium", "large"))
    assert small.mlp_widths < medium.mlp_widths < large.mlp_widths
    assert small.convolution_widths < medium.convolution_widths < large.convolution_widths
    with pytest.raises(ValueError, match="capacity must be one of"):
        capacity_spec("extra-large")


def test_neural_baseline_parameter_count_increases_with_capacity() -> None:
    pytest.importorskip("torch")
    from models.complex_cnn import build_complex_cnn
    from models.network import build_domain_mlp
    from models.real_cnn import build_real_cnn

    for builder, args in (
        (build_domain_mlp, (128,)),
        (build_real_cnn, (4,)),
        (build_complex_cnn, (2,)),
    ):
        counts = [
            sum(parameter.numel() for parameter in builder(*args, capacity=name).parameters())
            for name in ("small", "medium", "large")
        ]
        assert counts[0] < counts[1] < counts[2]


def test_sklearn_mlp_uses_named_capacity() -> None:
    from models.classical import make_estimator

    for capacity in ("small", "medium", "large"):
        estimator = make_estimator("mlp", seed=5, capacity=capacity)
        assert (
            estimator.named_steps["mlpclassifier"].hidden_layer_sizes
            == capacity_spec(capacity).mlp_widths
        )

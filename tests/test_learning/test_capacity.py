from __future__ import annotations

import pytest
from models.capacity import capacity_spec, matched_capacity_names


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


def test_shape_matched_neural_capacities_are_within_declared_parameter_tolerance() -> None:
    pytest.importorskip("torch")
    from models.classical import make_estimator
    from models.complex_cnn import build_complex_cnn
    from models.real_cnn import build_real_cnn

    links, tones = 2, 64
    profile = matched_capacity_names(links=links, subcarriers=tones)
    target = profile["target_parameters"]
    assert isinstance(target, int)

    mlp_capacity = str(profile["mlp_capacity"])
    mlp = make_estimator("mlp", seed=5, capacity=mlp_capacity)
    assert mlp.named_steps["mlpclassifier"].hidden_layer_sizes == (128, 64, 32)
    mlp_count = sum(
        (left + 1) * right
        for left, right in zip((2 * links * tones, 128, 64, 32), (128, 64, 32, 2))
    )

    real = build_real_cnn(2 * links, capacity=str(profile["real_cnn_capacity"]))
    complex_model = build_complex_cnn(links, capacity=str(profile["complex_cnn_capacity"]))
    real_count = sum(parameter.numel() for parameter in real.parameters())
    complex_count = sum(parameter.numel() for parameter in complex_model.parameters())
    assert mlp_count == target
    for count in (real_count, complex_count):
        assert abs(count - target) / target <= 0.1


def test_shape_matched_cnn_checkpoints_reconstruct_the_resolved_widths(tmp_path) -> None:
    pytest.importorskip("torch")
    import numpy as np
    from models.capacity import matched_capacity_names
    from models.complex_cnn import ComplexCNNPredictor, build_complex_cnn, load_complex_cnn
    from models.real_cnn import RealCNNPredictor, build_real_cnn, load_real_cnn

    profile = matched_capacity_names(links=1, subcarriers=64)
    real_capacity = str(profile["real_cnn_capacity"])
    real_model = build_real_cnn(2, real_capacity)
    real = RealCNNPredictor(
        real_model,
        np.zeros((1, 2, 1), dtype=np.float32),
        np.ones((1, 2, 1), dtype=np.float32),
        "cpu",
        capacity=real_capacity,
    )
    real_path = tmp_path / "real.pt"
    real.save(real_path, threshold=0.5)
    real_loaded = load_real_cnn(real_path)
    assert sum(parameter.numel() for parameter in real.model.parameters()) == sum(
        parameter.numel() for parameter in real_loaded.model.parameters()
    )

    complex_capacity = str(profile["complex_cnn_capacity"])
    complex_model = build_complex_cnn(1, complex_capacity)
    complex_predictor = ComplexCNNPredictor(
        complex_model,
        np.ones(1, dtype=np.float32),
        "cpu",
        capacity=complex_capacity,
    )
    complex_path = tmp_path / "complex.pt"
    complex_predictor.save(complex_path, threshold=0.5)
    complex_loaded = load_complex_cnn(complex_path)
    assert sum(parameter.numel() for parameter in complex_predictor.model.parameters()) == sum(
        parameter.numel() for parameter in complex_loaded.model.parameters()
    )

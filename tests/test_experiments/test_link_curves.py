import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
from experiments.link_curves import (
    run_tdl_pusch_bler_curve,
    run_tdl_pusch_rx_diversity_curve,
    run_tdl_pusch_time_bler_curve,
    torch_cuda_available,
    wilson_score_interval_95,
)


def test_wilson_interval_has_finite_upper_bound_for_zero_observed_errors():
    lower, upper = wilson_score_interval_95(0, 100)

    assert lower == 0.0
    assert 0.03 < upper < 0.04


def test_wilson_interval_is_symmetric_at_half_error_rate():
    lower, upper = wilson_score_interval_95(50, 100)

    assert lower == pytest.approx(1.0 - upper)
    assert lower < 0.5 < upper


def test_wilson_interval_handles_all_errors_and_rejects_boolean_total():
    lower, upper = wilson_score_interval_95(10, 10)
    assert lower > 0.7
    assert upper == 1.0
    with pytest.raises(TypeError, match="total must be an integer"):
        wilson_score_interval_95(1, True)


@pytest.mark.parametrize(
    ("errors", "total", "error"),
    [
        (True, 10, TypeError),
        (1.0, 10, TypeError),
        (-1, 10, ValueError),
        (11, 10, ValueError),
        (0, 0, ValueError),
    ],
)
def test_wilson_interval_rejects_invalid_counts(errors, total, error):
    with pytest.raises(error):
        wilson_score_interval_95(errors, total)


def test_link_curve_rejects_empty_snr_grid_before_loading_native_dependencies():
    with pytest.raises(ValueError, match="snr_db"):
        run_tdl_pusch_bler_curve(
            snr_db=[],
            frames_per_snr=1,
            batch_size=1,
            seed=1,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"frames_per_snr": True}, TypeError, "frames_per_snr must be an integer"),
        ({"batch_size": True}, TypeError, "batch_size must be an integer"),
        ({"frames_per_snr": 0}, ValueError, "must be positive"),
        ({"seed": -1}, ValueError, "seed must be a nonnegative integer"),
        ({"profile": "F"}, ValueError, "one of TDL-A through TDL-E"),
        ({"delay_spread_s": 0.0}, ValueError, "delay_spread_s must be finite"),
        ({"delay_spread_s": float("inf")}, ValueError, "delay_spread_s must be finite"),
        ({"carrier_frequency_hz": 0}, ValueError, "carrier_frequency_hz must be finite"),
        ({"speed_mps": -1}, ValueError, "speed_mps must be finite"),
    ],
)
def test_link_curve_rejects_invalid_tdl_configuration_before_loading_dependencies(
    overrides, error, message
):
    arguments = {"snr_db": (0.0,), "frames_per_snr": 1, "batch_size": 1, "seed": 1}
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        run_tdl_pusch_bler_curve(**arguments)


def test_resource_grid_curve_reports_missing_torch_install(monkeypatch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mocked absent torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    with pytest.raises(RuntimeError, match="Install PyTorch"):
        run_tdl_pusch_bler_curve(snr_db=(0.0,), frames_per_snr=1, batch_size=1, seed=1)


def _mock_resource_grid_curve_dependencies(monkeypatch, *, mismatch_channel=False):
    torch = pytest.importorskip("torch")
    calls = []

    class Receiver:
        def __init__(self, config, *, device):
            self.config = config
            self.device = device

    class Decoder:
        def __init__(self, receiver):
            self.receiver = receiver

        def __call__(self, received_grid, noise_variance):
            count = received_grid.shape[0]
            return (
                torch.zeros((count, 2), dtype=torch.int64),
                torch.ones((count, 1), dtype=torch.bool),
            )

    def generate(receiver, *, model, batch_size, seed, noise_variance, **kwargs):
        calls.append((model, batch_size, seed, noise_variance))
        channel_value = noise_variance if mismatch_channel else 1.0
        return SimpleNamespace(
            transmit_bits=torch.ones((batch_size, 2), dtype=torch.int64),
            channel_truth=torch.full((batch_size, 1), channel_value, dtype=torch.complex64),
            received_grid=torch.zeros((batch_size, 1), dtype=torch.complex64),
            noise_variance=torch.full((batch_size,), noise_variance),
        )

    receiver_module = ModuleType("integrations.sionna.nr_pusch")
    receiver_module.SionnaPUSCHLSReceiver = Receiver
    link_module = ModuleType("integrations.sionna.tdl_pusch")
    link_module.SionnaPUSCHLinkDecoder = Decoder
    link_module.generate_tdl_pusch_batch = generate
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    monkeypatch.setitem(sys.modules, link_module.__name__, link_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return calls


def test_resource_grid_curve_aggregates_batches_and_reuses_realizations(monkeypatch):
    calls = _mock_resource_grid_curve_dependencies(monkeypatch)

    report = run_tdl_pusch_bler_curve(
        snr_db=(0.0, 10.0),
        frames_per_snr=3,
        batch_size=2,
        seed=17,
        profile="c",
    )

    assert [call[:3] for call in calls] == [
        ("C", 2, 17),
        ("C", 1, 18),
        ("C", 2, 17),
        ("C", 1, 18),
    ]
    assert report["evidence_grade"] == "SIMULATION_ONLY_UNVALIDATED"
    assert report["configuration"]["batch_seeds_shared_across_snr"] == [17, 18]
    assert [point["frames"] for point in report["points"]] == [3, 3]
    assert [point["bits"] for point in report["points"]] == [6, 6]
    assert [point["bler"] for point in report["points"]] == [0.0, 0.0]
    assert [point["ber"] for point in report["points"]] == [1.0, 1.0]
    assert all(
        check["transmit_bits_match_across_snr"] and check["channel_truth_allclose_across_snr"]
        for check in report["configuration"]["matched_realization_checks"]
    )


def test_resource_grid_curve_rejects_a_mismatched_channel_realization(monkeypatch):
    _mock_resource_grid_curve_dependencies(monkeypatch, mismatch_channel=True)

    with pytest.raises(RuntimeError, match="did not reproduce bits/channel truth"):
        run_tdl_pusch_bler_curve(
            snr_db=(0.0, 10.0),
            frames_per_snr=1,
            batch_size=1,
            seed=17,
        )


def test_sample_domain_curve_aggregates_seeded_batches_with_mock_link(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class Receiver:
        def __init__(self, config, *, device):
            self.config = config
            self.device = device

    class Link:
        metadata = {"cp_covers_channel_memory": False, "channel_model": "test-double"}

        def __init__(self, receiver, **kwargs):
            self.receiver = receiver
            self.kwargs = kwargs

        def transmit_channel(self, count, *, seed):
            calls.append((count, seed))
            return (
                torch.full((count, 2), 2 + 0j, dtype=torch.complex64),
                torch.ones((count, 2), dtype=torch.int64),
                torch.ones((count, 1, 1, 2), dtype=torch.complex64),
                torch.full((count, 1), 1 + 0j, dtype=torch.complex64),
            )

        def decode(self, received, noise_variance):
            count = received.shape[0]
            return (
                torch.ones((count, 2), dtype=torch.int64),
                torch.ones((count, 1), dtype=torch.bool),
            )

    receiver_module = ModuleType("integrations.sionna.nr_pusch")
    receiver_module.SionnaPUSCHLSReceiver = Receiver
    link_module = ModuleType("integrations.sionna.tdl_pusch")
    link_module.SionnaTDLTimePUSCHLink = Link

    def seeded_awgn(shape, *, device, **kwargs):
        return torch.zeros(shape, dtype=torch.complex64, device=device)

    link_module.seeded_complex_awgn = seeded_awgn
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    monkeypatch.setitem(sys.modules, link_module.__name__, link_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = run_tdl_pusch_time_bler_curve(
        snr_db=(0.0, 10.0), frames_per_snr=3, batch_size=2, seed=23
    )

    assert calls == [(2, 23), (1, 24)]
    assert report["evidence_grade"] == "SIMULATION_ONLY_UNVALIDATED"
    assert report["configuration"]["device"] == "cpu"
    assert report["configuration"]["batch_seeds_shared_across_snr"] == [23, 24]
    assert report["limitations"][2].startswith("The discrete channel memory exceeds")
    for point in report["points"]:
        assert point["frames"] == 3
        assert point["bits"] == 6
        assert point["bler"] == point["ber"] == 0.0
        assert point["frame_cluster_ids"] == ["23", "23", "24"]
        assert point["mean_transmitted_tb_bits_per_codeword"] == 2


@pytest.mark.parametrize("rx_correlation", [0.0, 0.5])
def test_rx_diversity_curve_compares_antennas_from_one_mocked_channel_draw(
    monkeypatch, rx_correlation
):
    torch = pytest.importorskip("torch")
    calls = []

    class Receiver:
        def __init__(self, config, *, device):
            self.config = config
            self.device = device

    class Link:
        metadata = {"channel_model": "test-double"}

        def __init__(self, receiver, **kwargs):
            self.receiver = receiver
            self.kwargs = kwargs

        def transmit_channel(self, count, *, seed):
            calls.append((count, seed))
            clean = torch.empty((count, 1, 2, 3), dtype=torch.complex64)
            clean[:, :, 0, :] = 1 + 0j
            clean[:, :, 1, :] = 2 + 0j
            channel_truth = clean.unsqueeze(-1).expand(count, 1, 2, 3, 2).clone()
            return (
                torch.full((count, 2), 3 + 0j, dtype=torch.complex64),
                torch.ones((count, 2), dtype=torch.int64),
                clean,
                channel_truth,
            )

        def decode(self, received, noise_variance):
            count = received.shape[0]
            succeeds = received.shape[2] == 2
            return (
                torch.zeros((count, 2), dtype=torch.int64),
                torch.full((count, 1), succeeds, dtype=torch.bool),
            )

    receiver_module = ModuleType("integrations.sionna.nr_pusch")
    receiver_module.SionnaPUSCHLSReceiver = Receiver
    link_module = ModuleType("integrations.sionna.tdl_pusch")
    link_module.SionnaTDLTimePUSCHLink = Link
    link_module.seeded_complex_awgn = lambda shape, **kwargs: torch.zeros(
        shape, dtype=torch.complex64
    )
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    monkeypatch.setitem(sys.modules, link_module.__name__, link_module)
    monkeypatch.setattr("experiments.link_curves.torch_cuda_available", lambda: True)

    report = run_tdl_pusch_rx_diversity_curve(
        snr_db=(0.0, 10.0),
        frames_per_snr=3,
        batch_size=2,
        seed=31,
        rx_correlation=rx_correlation,
    )

    assert calls == [(2, 31), (1, 32)]
    assert report["configuration"]["rx_correlation_coefficient"] == rx_correlation
    assert report["configuration"]["matched_realization_checks"][0][
        "single_rx_is_first_antenna_of_two_rx_draw"
    ]
    correlation_assumption = report["configuration"]["receive_spatial_correlation_assumption"]
    if rx_correlation == 0.0:
        assert correlation_assumption.startswith("uncorrelated Sionna TDL branches")
    else:
        assert correlation_assumption.startswith("post-sampled Cholesky transform")
    for point in report["points"]:
        assert point["frames"] == 3
        assert point["single_rx"]["bler_errors"] == 3
        assert point["two_rx"]["bler_errors"] == 0
        assert point["single_rx"]["bit_errors"] == point["two_rx"]["bit_errors"] == 6
        expected_second_power = 4.0 if rx_correlation == 0.0 else (0.5 + (0.75**0.5) * 2.0) ** 2
        assert point["mean_clean_receive_power_by_antenna"] == pytest.approx(
            [1.0, expected_second_power]
        )
        assert 0.0 <= point["paired_comparison"]["mcnemar_exact_two_sided_p_value_holm"] <= 1.0


def test_rx_diversity_curve_reports_cuda_unavailable_after_input_validation(monkeypatch):
    monkeypatch.setattr("experiments.link_curves.torch_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        run_tdl_pusch_rx_diversity_curve(snr_db=(0.0,), frames_per_snr=1, batch_size=1, seed=1)


def test_cuda_probe_returns_false_when_torch_is_not_installed(monkeypatch):
    from experiments.link_curves import torch_cuda_available

    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mocked absent torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    assert torch_cuda_available() is False


def test_cuda_probe_matches_the_installed_torch_runtime():
    torch = pytest.importorskip("torch")
    assert torch_cuda_available() is torch.cuda.is_available()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"frames_per_snr": True}, TypeError, "frames_per_snr must be an integer"),
        ({"batch_size": True}, TypeError, "batch_size must be an integer"),
        ({"frames_per_snr": 0}, ValueError, "must be positive"),
        ({"seed": -1}, ValueError, "seed must be a nonnegative integer"),
        ({"profile": "F"}, ValueError, "one of TDL-A through TDL-E"),
        ({"delay_spread_s": 0.0}, ValueError, "delay_spread_s must be finite"),
        ({"carrier_frequency_hz": 0.0}, ValueError, "carrier_frequency_hz must be finite"),
        ({"speed_mps": -1.0}, ValueError, "speed_mps must be finite"),
        ({"rx_correlation": 1.0}, ValueError, "strictly between -1 and 1"),
    ],
)
def test_rx_diversity_curve_rejects_invalid_configuration_before_cuda_check(
    overrides, error, message
):
    arguments = {"snr_db": (0.0,), "frames_per_snr": 1, "batch_size": 1, "seed": 1}
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        run_tdl_pusch_rx_diversity_curve(**arguments)


def test_rx_diversity_curve_rejects_seed_without_room_for_derived_seeds(monkeypatch):
    pytest.importorskip("torch")
    receiver_module = ModuleType("integrations.sionna.nr_pusch")
    receiver_module.SionnaPUSCHLSReceiver = lambda config, *, device: object()
    link_module = ModuleType("integrations.sionna.tdl_pusch")
    link_module.SionnaTDLTimePUSCHLink = lambda receiver, **kwargs: object()
    link_module.seeded_complex_awgn = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    monkeypatch.setitem(sys.modules, link_module.__name__, link_module)
    monkeypatch.setattr("experiments.link_curves.torch_cuda_available", lambda: True)
    with pytest.raises(ValueError, match="leave room for derived channel and noise seeds"):
        run_tdl_pusch_rx_diversity_curve(
            snr_db=(0.0,), frames_per_snr=1, batch_size=1, seed=2**63 - 2
        )


def test_sample_domain_link_curve_rejects_empty_snr_grid_before_loading_dependencies():
    with pytest.raises(ValueError, match="snr_db"):
        run_tdl_pusch_time_bler_curve(
            snr_db=[],
            frames_per_snr=1,
            batch_size=1,
            seed=1,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"frames_per_snr": True}, TypeError, "frames_per_snr must be an integer"),
        ({"batch_size": True}, TypeError, "batch_size must be an integer"),
        ({"frames_per_snr": 0}, ValueError, "must be positive"),
        ({"seed": -1}, ValueError, "seed must be a nonnegative integer"),
        ({"profile": "F"}, ValueError, "one of TDL-A through TDL-E"),
        ({"delay_spread_s": 0.0}, ValueError, "delay_spread_s must be finite"),
        ({"carrier_frequency_hz": 0.0}, ValueError, "carrier_frequency_hz must be finite"),
        ({"speed_mps": -1.0}, ValueError, "speed_mps must be finite"),
        ({"channel_estimator_kind": "unknown"}, ValueError, "channel_estimator_kind"),
        (
            {"channel_estimator_kind": "sionna_ls", "estimator_min_delay_samples": 0},
            ValueError,
            "only valid for the finite-delay DFT estimator",
        ),
        (
            {
                "channel_estimator_kind": "finite_delay_dft",
                "estimator_min_delay_samples": 2,
                "estimator_max_delay_samples": 1,
            },
            ValueError,
            "must not exceed",
        ),
        ({"device": "mps"}, ValueError, "device must be cpu, cuda"),
    ],
)
def test_sample_domain_curve_rejects_invalid_configuration_before_link_creation(
    overrides, error, message
):
    arguments = {"snr_db": (0.0,), "frames_per_snr": 1, "batch_size": 1, "seed": 1}
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        run_tdl_pusch_time_bler_curve(**arguments)


def test_sample_domain_curve_rejects_seed_without_room_for_derived_seeds(monkeypatch):
    torch = pytest.importorskip("torch")
    receiver_module = ModuleType("integrations.sionna.nr_pusch")
    receiver_module.SionnaPUSCHLSReceiver = lambda config, *, device: object()
    link_module = ModuleType("integrations.sionna.tdl_pusch")
    link_module.SionnaTDLTimePUSCHLink = lambda receiver, **kwargs: object()
    link_module.seeded_complex_awgn = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    monkeypatch.setitem(sys.modules, link_module.__name__, link_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="leave room for derived channel and noise seeds"):
        run_tdl_pusch_time_bler_curve(snr_db=(0.0,), frames_per_snr=1, batch_size=1, seed=2**63 - 2)


def test_rx_diversity_curve_rejects_empty_snr_grid_before_loading_dependencies():
    with pytest.raises(ValueError, match="snr_db"):
        run_tdl_pusch_rx_diversity_curve(
            snr_db=[],
            frames_per_snr=1,
            batch_size=1,
            seed=1,
        )


@pytest.mark.parametrize("position", [-1, 4, True])
def test_sample_domain_link_curve_rejects_invalid_dmrs_position(position):
    expected = TypeError if isinstance(position, bool) else ValueError
    with pytest.raises(expected, match="dmrs_additional_position"):
        run_tdl_pusch_time_bler_curve(
            snr_db=(10.0,),
            frames_per_snr=1,
            batch_size=1,
            seed=1,
            dmrs_additional_position=position,
        )


@pytest.mark.parametrize(
    ("num_rx_ant", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_sample_domain_link_curve_rejects_invalid_receive_antenna_count(num_rx_ant, error):
    with pytest.raises(error, match="num_rx_ant"):
        run_tdl_pusch_time_bler_curve(
            snr_db=(10.0,),
            frames_per_snr=1,
            batch_size=1,
            seed=1,
            num_rx_ant=num_rx_ant,
        )

"""End-to-end CPU PUSCH checks for G5 ordered multipath impairments."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

import experiments.g5_ordered_impairments_receiver as experiment  # noqa: E402
from experiments.g5_ordered_impairments_receiver import (  # noqa: E402
    run_ordered_impairment_probe,
)


@pytest.mark.native_sionna
def test_known_ordered_impairments_match_reference_and_recover_pusch_payloads():
    """A small independent-seed fixture exercises the frozen acceptance contract."""
    report = run_ordered_impairment_probe(payload_seeds=(74_101, 74_102), blocks_per_seed=8)

    assert report["acceptance"]["pass"] is True
    assert report["acceptance"]["observed"]["max_abs_timing_error_samples"] == 0
    assert report["acceptance"]["observed"]["max_samplewise_reference_absolute_error"] < 2e-5
    assert report["acceptance"]["observed"]["min_wrong_order_rmse"] > 1e-4
    assert report["paired_outcomes"]["clean"]["crc_failures"] == 0
    criteria = report["acceptance"]["criteria_frozen_before_run"]
    assert criteria["p95_abs_cfo_error_hz"] == 250.0
    assert criteria["max_abs_cfo_error_hz"] == 350.0
    assert criteria["max_impaired_corrected_bler"] == 0.10


def test_probe_restores_torch_thread_count_when_probe_raises(monkeypatch):
    original_threads = torch.get_num_threads()

    def fail_probe(**kwargs):
        del kwargs
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(experiment, "_run_ordered_impairment_probe", fail_probe)
    with pytest.raises(RuntimeError, match="synthetic probe failure"):
        run_ordered_impairment_probe(payload_seeds=(1,), blocks_per_seed=1)

    assert torch.get_num_threads() == original_threads

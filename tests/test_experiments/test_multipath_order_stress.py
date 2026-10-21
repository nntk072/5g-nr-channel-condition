from experiments.multipath_order_stress import run_multipath_order_stress


def test_unknown_path_order_stress_is_deterministic_and_records_limitations():
    kwargs = {
        "seed": 1802,
        "trials_per_cell": 3,
        "tone_count": 16,
        "bandwidths_hz": (5e6,),
        "snr_db_values": (10.0,),
        "path_orders": (0, 1, 2),
        "maximum_delay_s": 250e-9,
        "delay_grid_size": 65,
    }

    first = run_multipath_order_stress(**kwargs)
    second = run_multipath_order_stress(**kwargs)

    assert first == second
    assert first["evidence_grade"] == "ANALYTICAL_SYNTHETIC_STRESS_ONLY"
    assert len(first["cases"]) == 3
    assert {case["true_path_order"] for case in first["cases"]} == {0, 1, 2}
    assert all(case["trials"] == 3 for case in first["cases"])
    assert "receiver channel-estimation error" in first["limitations"][0]

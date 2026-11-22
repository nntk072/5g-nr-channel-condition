"""Admission and independent table checks for fixed TDL-A30/D30 in FR2.

The gate's profile matrix declares fixed TS 38.101-4 A30 and D30 available in
FR2, but existing standardized-record admission fixtures exercise those two
profiles in FR1. This native test checks only those missing 28-GHz admission
combinations and independently compares retained absolute tap delays and
normalized powers with the hash-pinned TS 38.101-4 v19.2.2 resources.
It does not qualify other bands, bandwidths, fading cases, or physical channel
validity.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from datasets import StandardizedFamily, standardized_run_from_channel  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models  # noqa: E402

_FIXED_PROFILE_SHA256 = {
    "A30": "827bb27528cbf9bf48ff38fdfbff086b8344e282927daf6e4f3aa4357ada7838",
    "D30": "6da72f15e0da4fb972cf6a2c9c4ec2d2c1330f598f15e71c83cfbb324b59353c",
}


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ("A30", "D30"))
def test_fixed_tdl_a30_d30_are_admitted_in_fr2_against_pinned_profile_table(
    model: str,
) -> None:
    resource = models.fixed_tdl_parameter_file(f"TDL-{model}.json")
    resource_bytes = resource.read_bytes()
    source_digest = hashlib.sha256(resource_bytes).hexdigest()
    assert source_digest == _FIXED_PROFILE_SHA256[model]
    parameters = json.loads(resource_bytes)

    channel = generate_tdl_frequency_response(
        model=model,
        seed=13_201 if model == "A30" else 13_202,
        carrier_frequency_hz=28.0e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        delay_spread_s=None,
        spec_version="19.2",
        device="cpu",
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.TDL)
    record = run.records[0]

    # The reference is read from the fixed, version-pinned profile resource,
    # not reconstructed from fields emitted by the adapter or admitted record.
    table_delays_s = torch.as_tensor(parameters["delays"], dtype=torch.float64) * 1e-9
    table_powers = torch.pow(
        10.0,
        torch.as_tensor(parameters["powers"], dtype=torch.float64) / 10.0,
    )
    if parameters["los"]:
        expected_delays_s = table_delays_s[1:]
        expected_powers = torch.cat((table_powers[:1] + table_powers[1:2], table_powers[2:]))
    else:
        expected_delays_s = table_delays_s
        expected_powers = table_powers
    expected_powers /= torch.sum(expected_powers)

    assert record.profile_model == model
    assert record.metadata["carrier_frequency_hz"] == 28.0e9
    assert record.metadata["fixed_profile_spec_version"] == "19.2.2"
    assert record.metadata["delay_profile_semantics"] == "fixed_ts_38_101_4_absolute_tap_delays"
    torch.testing.assert_close(
        torch.as_tensor(record.tau_s[0, 0, :].copy(), dtype=torch.float64),
        expected_delays_s,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        torch.as_tensor(record.metadata["mean_tap_powers"], dtype=torch.float64),
        expected_powers,
        rtol=1e-6,
        atol=1e-7,
    )

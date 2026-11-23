"""UMCD physical-channel views, frequency receiver observations and phase-gauge rules."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhaseComparisonDisposition,
    PhysicalChannelView,
    phase_comparison_disposition,
)
from datasets.episode_store import (
    _channel_payload,
    read_episode_bundle,
    read_frequency_model_inputs,
    read_receiver_inference_records,
    read_receiver_model_inputs,
    write_episode_bundle,
)
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from tests.test_datasets.umcd_schema_support import (
    _bundle,
    _episode,
    _provenance_record,
    _receiver_binding,
)


def test_physical_channel_view_requires_named_axes_and_explicit_normalization():
    episode = _episode()
    arguments = {
        "view_id": "oracle-cfr-1",
        "link_id": "gNB-1-to-UE-1",
        "channel": np.ones((1, 1, 2, 3), dtype=np.complex64),
        "frame_times_s": episode.frame_times_s,
        "rx_port_ids": ("UE-V",),
        "tx_port_ids": ("gNB-V", "gNB-H"),
        "carrier_hz": episode.carrier_hz,
        "subcarrier_offsets_hz": episode.subcarrier_offsets_hz,
        "delay_reference": ChannelDelayReference.ABSOLUTE_PROPAGATION,
        "power_scale": ChannelPowerScale.ABSOLUTE,
    }
    channel = PhysicalChannelView(**arguments)
    channel.validate_for(episode)
    assert channel.axes == ("frame", "rx_port", "tx_port", "subcarrier")
    assert not channel.channel.flags.writeable
    assert channel.valid_mask is not None and channel.valid_mask.all()
    assert not channel.valid_mask.flags.writeable
    with pytest.raises(ValueError, match="phase_gauge_id"):
        PhysicalChannelView(**{**arguments, "phase_gauge_id": " cal-chain "})
    with pytest.raises(ValueError, match="masked channel elements"):
        PhysicalChannelView(
            **{
                **arguments,
                "valid_mask": np.zeros_like(arguments["channel"], dtype=np.bool_),
            }
        )
    invalid_zero_channel = PhysicalChannelView(
        **{
            **arguments,
            "channel": np.zeros_like(arguments["channel"]),
            "valid_mask": np.zeros_like(arguments["channel"], dtype=np.bool_),
        }
    )
    assert invalid_zero_channel.valid_mask is not None
    assert not invalid_zero_channel.valid_mask.any()
    with pytest.raises(ValueError, match="normalization_recipe"):
        PhysicalChannelView(
            **{
                **arguments,
                "power_scale": ChannelPowerScale.NORMALIZED,
            }
        )


def test_phase_gauge_comparison_requires_matching_reference_and_physical_grid():
    episode = _episode()
    arguments = {
        "view_id": "phase-view-a",
        "link_id": "gNB-1-to-UE-1",
        "channel": np.ones((1, 1, 2, 3), dtype=np.complex64),
        "frame_times_s": episode.frame_times_s,
        "rx_port_ids": ("UE-V",),
        "tx_port_ids": ("gNB-V", "gNB-H"),
        "carrier_hz": episode.carrier_hz,
        "subcarrier_offsets_hz": episode.subcarrier_offsets_hz,
        "delay_reference": ChannelDelayReference.ABSOLUTE_PROPAGATION,
        "power_scale": ChannelPowerScale.ABSOLUTE,
    }
    no_gauge = PhysicalChannelView(**arguments)
    same_gauge = PhysicalChannelView(**{**arguments, "phase_gauge_id": "cal-chain-7"})
    matching_gauge = PhysicalChannelView(
        **{**arguments, "view_id": "phase-view-b", "phase_gauge_id": "cal-chain-7"}
    )
    different_gauge = PhysicalChannelView(**{**arguments, "phase_gauge_id": "cal-chain-8"})
    other_carrier = PhysicalChannelView(
        **{**arguments, "phase_gauge_id": "cal-chain-7", "carrier_hz": 3.6e9}
    )

    assert (
        phase_comparison_disposition(no_gauge, same_gauge)
        is PhaseComparisonDisposition.REFERENCE_MISSING
    )
    assert (
        phase_comparison_disposition(same_gauge, different_gauge)
        is PhaseComparisonDisposition.GAUGE_MISMATCH
    )
    assert (
        phase_comparison_disposition(same_gauge, other_carrier)
        is PhaseComparisonDisposition.FREQUENCY_MISMATCH
    )
    assert (
        phase_comparison_disposition(same_gauge, matching_gauge)
        is PhaseComparisonDisposition.ELIGIBLE
    )


def test_phase_gauge_is_preserved_in_content_addressed_channel_layer(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    bundle = _bundle()
    channel = replace(bundle.channel_views[0], phase_gauge_id="cal-chain-7")
    bundle = replace(bundle, channel_views=(channel,))

    with np.load(io.BytesIO(_channel_payload(channel)), allow_pickle=False) as archive:
        layer_metadata = json.loads(archive["metadata_json"].tobytes())
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "umcd-physical-channel-layer.v1.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(layer_metadata)

    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.channel_views[0].phase_gauge_id == "cal-chain-7"


def test_physical_channel_view_rejects_coercible_timestamp_strings():
    channel = _bundle().channel_views[0]
    with pytest.raises(TypeError, match="real numeric timestamps"):
        replace(channel, frame_times_s=("0.0",))


def test_physical_channel_view_rejects_nonpositive_absolute_rf_frequency():
    channel = _bundle().channel_views[0]
    with pytest.raises(ValueError, match="positive RF frequencies"):
        replace(
            channel,
            carrier_hz=1.0,
            subcarrier_offsets_hz=np.asarray([-1.0, 0.0, 1.0]),
        )


@pytest.mark.parametrize(
    "frequencies",
    (
        np.asarray([True, True, True]),
        np.asarray(["-1", "0", "1"]),
        np.asarray([-1.0, 0.0, 1.0], dtype=object),
        np.asarray([-1.0 + 0j, 0.0 + 0j, 1.0 + 0j]),
    ),
)
def test_physical_channel_view_rejects_coercible_non_real_frequency_axes(frequencies):
    channel = _bundle().channel_views[0]
    with pytest.raises(TypeError, match="real numeric dtype"):
        replace(channel, subcarrier_offsets_hz=frequencies)


@pytest.mark.parametrize("carrier", (True, "3.5e9", 3.5e9 + 0j))
def test_physical_channel_view_rejects_non_real_carrier_scalars(carrier):
    channel = _bundle().channel_views[0]
    with pytest.raises(TypeError, match="real numeric scalar"):
        replace(channel, carrier_hz=carrier)


def test_frequency_receiver_observation_roundtrip_is_separate_and_inference_only(tmp_path):
    bundle = _bundle()
    episode = bundle.episode
    frequency_input = FrequencyDomainModelInputs(
        observed_cfr=np.full((1, 2, 3), 0.5 + 0.25j, dtype=np.complex64),
        noise_variance=0.05,
        snr_db=10.0,
        carrier_hz=episode.carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
    )
    linked = LinkedFrequencyDomainObservation(
        "gNB-1-to-UE-1",
        0,
        FrequencyDomainObservation("direct-cfr-1", frequency_input),
    )
    bundle = replace(bundle, receiver_observations=(linked,))

    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)
    inferred = read_frequency_model_inputs(tmp_path, stored)

    assert isinstance(loaded.receiver_observations[0], LinkedFrequencyDomainObservation)
    np.testing.assert_array_equal(inferred[0].observed_cfr, frequency_input.observed_cfr)
    np.testing.assert_array_equal(inferred[0].subcarrier_offsets_hz, episode.subcarrier_offsets_hz)
    assert inferred[0].noise_variance == 0.05
    assert not hasattr(inferred[0], "targets")
    with pytest.raises(ValueError, match="no NR receiver observations"):
        read_receiver_model_inputs(tmp_path, stored)


def test_frequency_receiver_observation_rejects_wrong_port_or_frequency_axes():
    bundle = _bundle()
    episode = bundle.episode
    bad_observation = FrequencyDomainObservation(
        "bad-cfr",
        FrequencyDomainModelInputs(
            observed_cfr=np.ones((1, 1, 4), dtype=np.complex64),
            noise_variance=0.0,
            snr_db=20.0,
            carrier_hz=episode.carrier_hz,
            subcarrier_offsets_hz=np.asarray([-45e3, -15e3, 0.0, 15e3]),
        ),
    )
    with pytest.raises(ValueError, match="axes do not match"):
        replace(
            bundle,
            receiver_observations=(
                LinkedFrequencyDomainObservation("gNB-1-to-UE-1", 0, bad_observation),
            ),
        )


def test_frequency_receiver_observation_rejects_nonpositive_absolute_rf_frequency():
    with pytest.raises(ValueError, match="positive RF frequencies"):
        FrequencyDomainModelInputs(
            observed_cfr=np.ones((1, 1, 3), dtype=np.complex64),
            noise_variance=0.0,
            snr_db=20.0,
            carrier_hz=1.0,
            subcarrier_offsets_hz=np.asarray([-1.0, 0.0, 1.0]),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("subcarrier_offsets_hz", np.asarray([-1.0, 0.0, 1.0], dtype=bool), "real numeric dtype"),
        ("subcarrier_offsets_hz", np.asarray(["-1", "0", "1"]), "real numeric dtype"),
        ("subcarrier_offsets_hz", np.asarray([-1.0, 0.0, 1.0], dtype=object), "real numeric dtype"),
        (
            "subcarrier_offsets_hz",
            np.asarray([-1.0 + 0j, 0.0 + 0j, 1.0 + 0j]),
            "real numeric dtype",
        ),
        ("noise_variance", True, "real numeric scalar"),
        ("snr_db", "10", "real numeric scalar"),
        ("carrier_hz", 3.5e9 + 0j, "real numeric scalar"),
    ],
)
def test_frequency_receiver_contract_rejects_coercible_numeric_aliases(field, value, message):
    model_inputs = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 1, 3), dtype=np.complex64),
        noise_variance=0.1,
        snr_db=10.0,
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.asarray([-15e3, 0.0, 15e3]),
    )
    with pytest.raises(TypeError, match=message):
        replace(model_inputs, **{field: value})


def test_frequency_observation_ids_must_be_canonical():
    model_inputs = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 1, 3), dtype=np.complex64),
        noise_variance=0.1,
        snr_db=10.0,
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.asarray([-15e3, 0.0, 15e3]),
    )
    with pytest.raises(ValueError, match="canonical string"):
        FrequencyDomainObservation(" cfr-1 ", model_inputs)
    observation = FrequencyDomainObservation("cfr-1", model_inputs)
    with pytest.raises(ValueError, match="canonical string"):
        LinkedFrequencyDomainObservation(" link-1 ", 0, observation)


def test_receiver_view_roundtrip_isolated_from_oracle_and_targets(tmp_path):
    bundle = replace(
        _bundle(),
        receiver_observations=(_receiver_binding(),),
        provenance_records=(_provenance_record(),),
    )
    stored = write_episode_bundle(tmp_path, bundle)

    records = read_receiver_inference_records(tmp_path, stored)
    assert len(records) == 1
    assert records[0].episode_id == bundle.episode.episode_id
    assert records[0].link_id == "gNB-1-to-UE-1"
    assert records[0].frame_index == 0
    assert records[0].sample_id == "sample-1"
    assert not hasattr(records[0].model_inputs, "sample_id")
    assert not hasattr(records[0].model_inputs, "targets")

    inference = read_receiver_model_inputs(tmp_path, stored)
    assert len(inference) == 1
    np.testing.assert_array_equal(
        inference[0].received_grid, bundle.receiver_observations[0].observation.received_grid
    )
    np.testing.assert_array_equal(
        inference[0].h_hat, bundle.receiver_observations[0].observation.channel_estimate.h_hat
    )
    assert inference[0].fft_bin_indices == (0, 1, 2, 3)
    assert not hasattr(inference[0], "sample_id")
    assert not hasattr(inference[0], "targets")

    loaded = read_episode_bundle(tmp_path, stored)
    assert loaded.receiver_observations[0].link_id == "gNB-1-to-UE-1"
    assert loaded.receiver_observations[0].observation.sample_id == "sample-1"
    assert loaded.provenance_records[0].record_id == "rt-solve-1"
    assert (
        loaded.provenance_records[0].material_config["building-concrete"]["scattering_coefficient"]
        == 0.0
    )
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert "receiver_observations" in {item["layer"] for item in manifest["layers"]}
    assert "provenance" in {item["layer"] for item in manifest["layers"]}

    oracle_ref = next(item for item in manifest["layers"] if item["layer"] == "oracle_paths")
    oracle_path = tmp_path / oracle_ref["path"]
    oracle_path.write_bytes(oracle_path.read_bytes() + b"tampered")
    # Receiver inference reads only the episode and receiver objects; the full
    # training bundle reader still checks every oracle/target object.
    assert len(read_receiver_model_inputs(tmp_path, stored)) == 1
    assert read_receiver_inference_records(tmp_path, stored)[0].sample_id == "sample-1"
    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(tmp_path, stored)

"""UMCD physical-channel views, frequency receiver observations and phase-gauge rules."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import datasets.episode_store.episode as episode_module
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
from datasets.time_axis import MultiSourceTimeAlignment, align_time_axes
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


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("view_id", "", ValueError, "view_id must be"),
        ("link_id", " padded ", ValueError, "link_id must be"),
        ("channel", np.ones((1, 2, 3)), ValueError, "channel must be finite complex"),
        ("channel", np.ones((1, 1, 2, 3)), ValueError, "channel must be finite complex"),
        (
            "channel",
            np.full((1, 1, 2, 3), np.inf + 0j),
            ValueError,
            "channel must be finite complex",
        ),
        ("rx_port_ids", (), ValueError, "must be non-empty strings"),
        ("tx_port_ids", ("gNB-H", "gNB-H"), ValueError, "port IDs must be unique"),
        ("rx_port_ids", (" padded ",), ValueError, "must be non-empty strings"),
        (
            "valid_mask",
            np.ones((1, 1, 2, 3), dtype=np.uint8),
            ValueError,
            "valid_mask must be boolean",
        ),
        (
            "valid_mask",
            np.ones((1, 1, 2, 2), dtype=np.bool_),
            ValueError,
            "valid_mask must be boolean",
        ),
        (
            "subcarrier_offsets_hz",
            np.asarray([-1.0, 0.0]),
            ValueError,
            "complete final channel axis",
        ),
        (
            "subcarrier_offsets_hz",
            np.asarray([-1.0, np.nan, 1.0]),
            ValueError,
            "complete final channel axis",
        ),
        ("subcarrier_offsets_hz", np.asarray([-1.0, -2.0, 0.0]), ValueError, "strictly increasing"),
        ("subcarrier_offsets_hz", np.asarray([-2.0, -1.0, 1.0]), ValueError, "include DC"),
        ("carrier_hz", 0.0, ValueError, "carrier_hz must be finite and positive"),
        ("delay_reference", "absolute", TypeError, "ChannelDelayReference"),
        ("power_scale", "absolute", TypeError, "ChannelPowerScale"),
        ("axes", ("rx_port", "frame", "tx_port", "subcarrier"), ValueError, "canonical named-axis"),
        ("source_time_alignment_sha256", "A" * 64, ValueError, "lowercase SHA-256"),
    ],
)
def test_physical_channel_view_rejects_invalid_axis_and_reference_contracts(
    field, value, error, message
):
    channel = _bundle().channel_views[0]
    with pytest.raises(error, match=message):
        replace(channel, **{field: value})


def test_physical_channel_view_rejects_channel_port_axis_count_mismatch():
    channel = _bundle().channel_views[0]
    with pytest.raises(ValueError, match="leading axes disagree"):
        replace(channel, rx_port_ids=("extra-port", "another-port"))


def test_physical_channel_view_requires_explicit_recipe_for_relative_delay():
    channel = _bundle().channel_views[0]
    with pytest.raises(ValueError, match="normalization_recipe"):
        replace(channel, delay_reference=ChannelDelayReference.RELATIVE)


def test_physical_channel_view_validation_binds_to_episode_contract():
    bundle = _bundle()
    channel = bundle.channel_views[0]
    altered_offsets = channel.subcarrier_offsets_hz.copy()
    altered_offsets[0] -= 1.0
    mismatches = (
        (
            replace(channel, frame_times_s=tuple(value + 1 for value in channel.frame_times_s)),
            "frame times",
        ),
        (replace(channel, carrier_hz=channel.carrier_hz + 1), "carrier"),
        (replace(channel, subcarrier_offsets_hz=altered_offsets), "frequency offsets"),
        (
            replace(channel, rx_port_ids=tuple("unbound-rx" for _ in channel.rx_port_ids)),
            "RX port IDs",
        ),
        (
            replace(
                channel,
                tx_port_ids=tuple(
                    f"unbound-tx-{index}" for index in range(len(channel.tx_port_ids))
                ),
            ),
            "TX port IDs",
        ),
    )
    for invalid, message in mismatches:
        with pytest.raises(ValueError, match=message):
            invalid.validate_for(bundle.episode)

    with pytest.raises(ValueError, match="source-time digest requires episode"):
        replace(channel, source_time_alignment_sha256="a" * 64).validate_for(bundle.episode)

    times = bundle.episode.frame_times_s
    alignment = MultiSourceTimeAlignment(
        reference_times_s=times,
        source_alignments=(("source-a", align_time_axes(times, times, tolerance_s=0.0)),),
    )
    aligned_episode = replace(bundle.episode, source_time_alignment=alignment)
    with pytest.raises(ValueError, match="must match the episode alignment report"):
        channel.validate_for(aligned_episode)
    replace(channel, source_time_alignment_sha256=alignment.sha256).validate_for(aligned_episode)


def test_phase_comparison_reports_axis_delay_and_power_mismatches():
    channel = _bundle().channel_views[0]
    calibrated = replace(channel, phase_gauge_id="shared-gauge")

    with pytest.raises(TypeError, match="two PhysicalChannelView"):
        phase_comparison_disposition(object(), calibrated)
    assert (
        phase_comparison_disposition(calibrated, replace(calibrated, link_id="other-link"))
        is PhaseComparisonDisposition.AXIS_MISMATCH
    )
    assert (
        phase_comparison_disposition(
            calibrated,
            replace(
                calibrated,
                delay_reference=ChannelDelayReference.RELATIVE,
                normalization_recipe="explicit-reference",
            ),
        )
        is PhaseComparisonDisposition.DELAY_REFERENCE_MISMATCH
    )
    assert (
        phase_comparison_disposition(
            calibrated,
            replace(
                calibrated,
                power_scale=ChannelPowerScale.NORMALIZED,
                normalization_recipe="explicit-reference",
            ),
        )
        is PhaseComparisonDisposition.POWER_REFERENCE_MISMATCH
    )


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


@pytest.mark.parametrize("change", ("carrier", "offsets"))
def test_frequency_receiver_observation_must_match_parent_grid(change: str) -> None:
    bundle = _bundle()
    episode = bundle.episode
    offsets = episode.subcarrier_offsets_hz
    carrier = episode.carrier_hz
    if change == "carrier":
        carrier += 1.0
    else:
        offsets = offsets * 2.0
    observation = FrequencyDomainObservation(
        "grid-mismatch",
        FrequencyDomainModelInputs(
            observed_cfr=np.ones((1, 2, 3), dtype=np.complex64),
            noise_variance=0.0,
            snr_db=20.0,
            carrier_hz=carrier,
            subcarrier_offsets_hz=offsets,
        ),
    )

    with pytest.raises(ValueError, match="frequency observation grid differs"):
        replace(
            bundle,
            receiver_observations=(
                LinkedFrequencyDomainObservation("gNB-1-to-UE-1", 0, observation),
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


def _stored_frequency_bundle(tmp_path):
    bundle = _bundle()
    episode = bundle.episode
    inputs = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 2, 3), dtype=np.complex64),
        noise_variance=0.05,
        snr_db=10.0,
        carrier_hz=episode.carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
    )
    binding = LinkedFrequencyDomainObservation(
        "gNB-1-to-UE-1", 0, FrequencyDomainObservation("frequency-sample", inputs)
    )
    stored = write_episode_bundle(tmp_path, replace(bundle, receiver_observations=(binding,)))
    return stored, binding


def test_frequency_reader_requires_one_episode_and_an_observation(tmp_path, monkeypatch):
    metadata = {"episode_id": "fixture"}
    monkeypatch.setattr(episode_module, "_validated_manifest", lambda *_args: (metadata, []))
    with pytest.raises(ValueError, match="one episode and at least one"):
        read_frequency_model_inputs(tmp_path, object())


def test_frequency_reader_rejects_episode_schema_and_identity_drift(tmp_path, monkeypatch):
    stored, _ = _stored_frequency_bundle(tmp_path)
    original = episode_module._read_object

    def bad_schema(base, reference, **kwargs):
        if reference["layer"] == "episode":
            return {"schema": "other"}, {}
        return original(base, reference, **kwargs)

    monkeypatch.setattr(episode_module, "_read_object", bad_schema)
    with pytest.raises(ValueError, match="unsupported episode layer schema"):
        read_frequency_model_inputs(tmp_path, stored)


def test_frequency_reader_rejects_manifest_episode_id_drift(tmp_path, monkeypatch):
    stored, _ = _stored_frequency_bundle(tmp_path)
    metadata, references = episode_module._validated_manifest(tmp_path, stored)
    metadata["episode_id"] = "different-episode"
    monkeypatch.setattr(
        episode_module, "_validated_manifest", lambda *_args: (metadata, references)
    )
    with pytest.raises(ValueError, match="manifest and episode payload IDs differ"):
        read_frequency_model_inputs(tmp_path, stored)


@pytest.mark.parametrize(
    ("link_id", "frame_index", "message"),
    [
        ("unlisted-link", 0, "unknown link or frame"),
        ("gNB-1-to-UE-1", 1, "unknown link or frame"),
    ],
)
def test_frequency_reader_rejects_unknown_link_or_frame(
    tmp_path, monkeypatch, link_id, frame_index, message
):
    stored, binding = _stored_frequency_bundle(tmp_path)
    monkeypatch.setattr(
        episode_module,
        "_read_frequency_receiver",
        lambda *_args: replace(binding, link_id=link_id, frame_index=frame_index),
    )
    with pytest.raises(ValueError, match=message):
        read_frequency_model_inputs(tmp_path, stored)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            FrequencyDomainModelInputs(
                observed_cfr=np.ones((1, 1, 3), dtype=np.complex64),
                noise_variance=0.05,
                snr_db=10.0,
                carrier_hz=3.5e9,
                subcarrier_offsets_hz=np.asarray([-30e3, 0.0, 30e3]),
            ),
            "axes do not match",
        ),
        (
            FrequencyDomainModelInputs(
                observed_cfr=np.ones((1, 2, 3), dtype=np.complex64),
                noise_variance=0.05,
                snr_db=10.0,
                carrier_hz=2.6e9,
                subcarrier_offsets_hz=np.asarray([-30e3, 0.0, 30e3]),
            ),
            "grid differs",
        ),
    ],
)
def test_frequency_reader_rejects_input_axis_or_grid_drift(tmp_path, monkeypatch, inputs, message):
    stored, binding = _stored_frequency_bundle(tmp_path)
    replacement = replace(
        binding, observation=FrequencyDomainObservation("frequency-sample", inputs)
    )
    monkeypatch.setattr(episode_module, "_read_frequency_receiver", lambda *_args: replacement)
    with pytest.raises(ValueError, match=message):
        read_frequency_model_inputs(tmp_path, stored)


def test_frequency_reader_rejects_duplicate_link_frame_observations(tmp_path, monkeypatch):
    stored, _ = _stored_frequency_bundle(tmp_path)
    metadata, references = episode_module._validated_manifest(tmp_path, stored)
    receiver = next(item for item in references if item["layer"] == "receiver_observations")
    monkeypatch.setattr(
        episode_module,
        "_validated_manifest",
        lambda *_args, **_kwargs: (metadata, references + [receiver]),
    )
    with pytest.raises(ValueError, match="duplicated for a link/frame"):
        read_frequency_model_inputs(tmp_path, stored)


def test_frequency_reader_rejects_duplicate_sample_ids(tmp_path, monkeypatch):
    stored, binding = _stored_frequency_bundle(tmp_path)
    metadata, references = episode_module._validated_manifest(tmp_path, stored)
    receiver = next(item for item in references if item["layer"] == "receiver_observations")
    monkeypatch.setattr(
        episode_module,
        "_validated_manifest",
        lambda *_args: (metadata, references + [receiver]),
    )
    original_read_episode = episode_module._read_episode
    monkeypatch.setattr(
        episode_module,
        "_read_episode",
        lambda *args: replace(original_read_episode(*args), frame_times_s=(0.0, 1.0)),
    )
    reads = iter((0, 1))
    monkeypatch.setattr(
        episode_module,
        "_read_frequency_receiver",
        lambda *_args: replace(binding, frame_index=next(reads)),
    )
    with pytest.raises(ValueError, match="sample IDs are duplicated"):
        read_frequency_model_inputs(tmp_path, stored)


def test_frequency_reader_skips_other_receiver_schemas(tmp_path, monkeypatch):
    stored, _ = _stored_frequency_bundle(tmp_path)
    original = episode_module._read_object

    def non_frequency_observation(base, reference, **kwargs):
        if reference["layer"] == "receiver_observations":
            return {"schema": "umcd.receiver_observation.v1"}, {}
        return original(base, reference, **kwargs)

    monkeypatch.setattr(episode_module, "_read_object", non_frequency_observation)
    with pytest.raises(ValueError, match="contains no direct noisy-CFR"):
        read_frequency_model_inputs(tmp_path, stored)


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

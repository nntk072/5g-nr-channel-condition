from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import (
    StandardizedChannelRecord,
    StandardizedFamily,
    read_standardized_learning_run,
    read_standardized_run,
    standardized_learning_example_from_record,
    standardized_learning_run_from_channel_run,
    standardized_run_from_channel,
    write_standardized_learning_run,
    write_standardized_run,
)


def _metadata(*, model: str = "A", los: bool = False) -> dict[str, object]:
    return {
        "sionna_version": "2.2.0",
        "spec_version": "19.2",
        "profile_proxy": model,
        "profile_model": model,
        "profile_los_component": los,
        "geometric_los_label": None,
        "carrier_frequency_hz": 3.5e9,
        "batch_size": 2,
        "num_time_steps": 2,
        "num_subcarriers": 8,
        "subcarrier_spacing_hz": 15_000.0,
        "delay_reference": "profile_excess_delay",
        "seed": 11,
        "nested_config": {"values": [1, 2]},
    }


def _channel(*, model: str = "A", los: bool = False) -> SimpleNamespace:
    h_path = np.zeros((2, 1, 1, 1, 1, 2, 2), dtype=np.complex64)
    h_path[:, 0, 0, 0, 0, 0, :] = (1.0 + 0.25j, 0.5 - 0.1j)
    h_path[:, 0, 0, 0, 0, 1, :] = (0.3 - 0.2j, 0.15 + 0.1j)
    tau_s = np.tile(np.asarray([0.0, 1.25e-6]), (2, 1, 1, 1)).astype(np.float64)
    offsets = (np.arange(8) - 4) * 15_000.0
    phase = np.exp(
        -2j
        * np.pi
        * tau_s[:, :, None, :, None, None, :, None]
        * offsets.reshape(1, 1, 1, 1, 1, 1, 1, -1)
    )
    cfr = np.sum(np.moveaxis(h_path, -2, -1)[..., None] * phase, axis=6)
    return SimpleNamespace(
        h_path=h_path,
        tau_s=tau_s,
        channel_frequency_response=cfr,
        metadata=_metadata(model=model, los=los),
    )


def test_channel_adapter_batch_round_trip_is_immutable_and_preserves_profile_semantics(
    tmp_path,
) -> None:
    run = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="tdl-a-fixture"
    )
    assert len(run.records) == 2
    record = run.records[0]
    assert record.family is StandardizedFamily.TDL
    assert record.profile_model == "A"
    assert record.profile_los_component is False
    assert record.metadata["geometric_los_label"] is None
    assert record.metadata["delay_reference"] == "profile_excess_delay"
    assert record.h_path.shape == (1, 1, 1, 1, 2, 2)
    assert record.cfr.shape == (1, 1, 1, 1, 2, 8)
    assert not record.h_path.flags.writeable
    with pytest.raises(ValueError):
        record.cfr[0, 0, 0, 0, 0, 0] = 0
    with pytest.raises(TypeError):
        record.metadata["seed"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        record.metadata["nested_config"]["values"] = (3,)  # type: ignore[index]
    with pytest.raises(AttributeError):
        record.metadata["nested_config"]["values"].append(3)

    stored = write_standardized_run(tmp_path, run)
    assert write_standardized_run(tmp_path, run) == stored
    restored = read_standardized_run(tmp_path, stored)
    assert [item.sample_id for item in restored.records] == [item.sample_id for item in run.records]
    for actual, expected in zip(restored.records, run.records, strict=True):
        np.testing.assert_array_equal(actual.h_path, expected.h_path)
        np.testing.assert_array_equal(actual.tau_s, expected.tau_s)
        np.testing.assert_allclose(actual.cfr, expected.cfr, rtol=0, atol=0)


def test_record_validation_rejects_geometric_los_and_inconsistent_cfr() -> None:
    with pytest.raises(ValueError, match="geometric LOS"):
        standardized_run_from_channel(
            SimpleNamespace(
                **{**vars(_channel()), "metadata": {**_metadata(), "geometric_los_label": "nlos"}}
            ),
            family=StandardizedFamily.TDL,
        )

    channel = _channel()
    channel.channel_frequency_response[0, 0, 0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="does not match"):
        standardized_run_from_channel(channel, family=StandardizedFamily.TDL)


def test_profile_los_is_not_geometric_los_and_family_category_is_checked() -> None:
    run = standardized_run_from_channel(
        _channel(model="D", los=True), family=StandardizedFamily.CDL
    )
    assert run.records[0].profile_los_component
    assert run.records[0].metadata["geometric_los_label"] is None

    record = run.records[0]
    with pytest.raises(ValueError, match="profile LOS category"):
        StandardizedChannelRecord(
            **{
                field: getattr(record, field)
                for field in (
                    "sample_id",
                    "run_id",
                    "sample_index",
                    "family",
                    "profile_model",
                    "h_path",
                    "tau_s",
                    "cfr",
                    "subcarrier_offsets_hz",
                    "metadata",
                )
            },
            profile_los_component=False,
        )


def test_record_content_hash_rejects_tampering(tmp_path) -> None:
    stored = write_standardized_run(
        tmp_path,
        standardized_run_from_channel(_channel(), family=StandardizedFamily.TDL),
    )
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    record_path = tmp_path / manifest["records"][0]["relative_path"]
    content = bytearray(record_path.read_bytes())
    content[-1] ^= 1
    record_path.write_bytes(content)
    with pytest.raises(ValueError, match="content hash"):
        read_standardized_run(tmp_path, stored)


def test_manifest_is_immutable_for_a_given_run_id(tmp_path) -> None:
    first = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="fixed-run"
    )
    write_standardized_run(tmp_path, first)
    altered_channel = _channel()
    altered_channel.metadata["seed"] = 12
    second = standardized_run_from_channel(
        altered_channel, family=StandardizedFamily.TDL, run_id="fixed-run"
    )
    with pytest.raises(FileExistsError):
        write_standardized_run(tmp_path, second)


def test_manifest_digest_is_bound_to_records(tmp_path) -> None:
    stored = write_standardized_run(
        tmp_path,
        standardized_run_from_channel(_channel(), family=StandardizedFamily.TDL),
    )
    manifest_path = tmp_path / stored.relative_path
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["sample_id"] = "forged"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        read_standardized_run(tmp_path, stored)


def test_record_hash_uses_content_addressed_filename(tmp_path) -> None:
    stored = write_standardized_run(
        tmp_path,
        standardized_run_from_channel(_channel(), family=StandardizedFamily.TDL),
    )
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    item = manifest["records"][0]
    assert (
        item["content_sha256"]
        == hashlib.sha256((tmp_path / item["relative_path"]).read_bytes()).hexdigest()
    )


def test_noisy_cfr_observation_and_profile_proxy_target_are_separate() -> None:
    run = standardized_run_from_channel(_channel(), family=StandardizedFamily.TDL)
    record = run.records[0]
    first = standardized_learning_example_from_record(record, snr_db=20.0, seed=404)
    replay = standardized_learning_example_from_record(record, snr_db=20.0, seed=404)
    other_noise = standardized_learning_example_from_record(record, snr_db=20.0, seed=405)

    np.testing.assert_array_equal(
        first.model_inputs().observed_cfr, replay.model_inputs().observed_cfr
    )
    assert not np.array_equal(
        first.model_inputs().observed_cfr, other_noise.model_inputs().observed_cfr
    )
    assert first.sample_id == record.sample_id
    assert first.targets["profile_los_proxy"].values.tolist() == [0]
    assert "not geometric LOS" in first.targets["profile_los_proxy"].definition.reference
    assert not hasattr(first.model_inputs(), "sample_id")
    assert not hasattr(first.model_inputs(), "targets")
    assert not hasattr(first.model_inputs(), "h_path")
    with pytest.raises(ValueError, match="outside"):
        standardized_learning_example_from_record(record, snr_db=20.0, seed=404, time_index=2)


def test_seeded_complex_awgn_matches_declared_noise_variance() -> None:
    run = standardized_run_from_channel(_channel(), family=StandardizedFamily.TDL)
    record = run.records[0]
    noiseless = np.asarray(record.cfr[0, :, 0, :, 0, :], dtype=np.complex128)
    noise_samples = []
    expected_variances = []
    for seed in range(240):
        example = standardized_learning_example_from_record(record, snr_db=12.0, seed=seed)
        noise_samples.append(example.model_inputs().observed_cfr.astype(np.complex128) - noiseless)
        expected_variances.append(example.model_inputs().noise_variance)

    empirical_variance = float(np.mean(np.abs(np.stack(noise_samples)) ** 2))
    expected_variance = float(np.mean(expected_variances))
    assert empirical_variance == pytest.approx(expected_variance, rel=0.08)


def test_learning_run_persists_observation_and_target_layers_separately(tmp_path) -> None:
    channels = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="tdl-a-learning"
    )
    learning = standardized_learning_run_from_channel_run(channels, snr_db=15.0, seed=7001)
    stored = write_standardized_learning_run(tmp_path, learning)
    restored = read_standardized_learning_run(tmp_path, stored)

    assert restored.run_id == channels.run_id
    assert [item.sample_id for item in restored.examples] == [
        item.sample_id for item in channels.records
    ]
    for original, actual in zip(learning.examples, restored.examples, strict=True):
        np.testing.assert_array_equal(
            actual.model_inputs().observed_cfr, original.model_inputs().observed_cfr
        )
        assert actual.targets["profile_los_proxy"].values.tolist() == [0]
        assert actual.observation_seed == original.observation_seed
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert all(set(item) == {"sample_id", "observation", "targets"} for item in manifest["records"])
    assert (tmp_path / manifest["records"][0]["observation"]["relative_path"]).is_file()
    assert (tmp_path / manifest["records"][0]["targets"]["relative_path"]).is_file()


def test_learning_store_rejects_tampered_target_layer(tmp_path) -> None:
    channels = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="tdl-a-tamper"
    )
    learning = standardized_learning_run_from_channel_run(channels, snr_db=12.0, seed=8001)
    stored = write_standardized_learning_run(tmp_path, learning)
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    target_path = tmp_path / manifest["records"][0]["targets"]["relative_path"]
    target_path.write_text(target_path.read_text().replace("profile_los_proxy", "forged"))
    with pytest.raises(ValueError, match="target content hash"):
        read_standardized_learning_run(tmp_path, stored)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("family", "model"), [(StandardizedFamily.TDL, "A"), (StandardizedFamily.CDL, "D")]
)
def test_pinned_sionna_adapter_outputs_round_trip_as_standardized_runs(
    tmp_path, family: StandardizedFamily, model: str
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    if family is StandardizedFamily.TDL:
        from integrations.sionna.tdl import generate_tdl_frequency_response

        channel = generate_tdl_frequency_response(
            model=model,
            seed=18,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000.0,
            num_subcarriers=16,
            delay_spread_s=100e-9,
            batch_size=2,
            num_time_steps=2,
            sampling_frequency_hz=15_000.0,
        )
    else:
        from integrations.sionna.cdl import generate_cdl_frequency_response

        channel = generate_cdl_frequency_response(
            model=model,
            seed=18,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000.0,
            num_subcarriers=16,
            delay_spread_s=100e-9,
            batch_size=2,
            num_time_steps=2,
            sampling_frequency_hz=15_000.0,
        )

    run = standardized_run_from_channel(channel, family=family)
    restored = read_standardized_run(tmp_path, write_standardized_run(tmp_path, run))
    assert len(restored.records) == 2
    assert all(record.family is family for record in restored.records)
    assert all(record.metadata["geometric_los_label"] is None for record in restored.records)
    assert all(record.h_path.shape[-1] == 2 for record in restored.records)

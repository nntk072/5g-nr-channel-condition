import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
    validate_canonical_target_contract,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _definition(version: str, *, semantic: str, units: str) -> TargetDefinition:
    provenance = {
        "profile_los_proxy.v2": (
            "Sionna standardized TDL/CDL profile model",
            "profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth",
            "single standardized profile category; false=0, true=1",
        ),
        "umcd.standardized.effective-cfr.v1": (
            "standardized Sionna TDL/CDL CFR adapter",
            "profile-relative excess-delay CFR resampled to this resource grid",
            "retain each effective subcarrier coefficient in receiver FFT-bin order; "
            "no antenna, symbol, or power averaging",
        ),
        "umcd.tdl-pusch.effective-cfr.v1": (
            "sionna_phy_ofdmchannel_return_channel",
            "OFDMChannel return_channel with the configured TDL profile, carrier, sample rate, "
            "and effective FFT-bin mapping",
            "retain each effective subcarrier coefficient; no antenna, symbol, or power averaging",
        ),
        "umcd.rt-trajectory.absolute-cfr.v1": (
            "sionna_rt_pathsolver",
            "Sionna RT full solver-valid path set evaluated with absolute propagation delays "
            "and configured carrier/subcarrier offsets",
            "coherent sum of all retained solver-valid paths per effective subcarrier and frame; "
            "no path-count or power normalization",
        ),
        "umcd.rt-trajectory.symbol-cfr.v1": (
            "sionna_rt_paths_cfr_doppler",
            "absolute-delay RT path channel evaluated at the declared NR OFDM symbol start timestamps",
            "coherent sum of retained solver-valid paths per effective subcarrier and OFDM symbol; "
            "no path or symbol averaging",
        ),
        "umcd.rt-trajectory.geometric-los.v1": (
            "mitsuba_segment_intersection_oracle",
            "independent finite-segment visibility query for the recorded TX/RX poses and scene geometry",
            "one independent direct-path visibility value per frame",
        ),
    }.get(
        version,
        (
            "canonical contract fixture",
            "frozen fixture reference",
            "preserve declared target elements",
        ),
    )
    return TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=provenance[0],
        reference=provenance[1],
        aggregation_rule=provenance[2],
        definition_version=version,
    )


def test_standardized_profile_los_target_is_single_binary_value() -> None:
    definition = _definition(
        "profile_los_proxy.v2",
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
    )
    target = TargetValue(
        entity_id="profile-a",
        definition=definition,
        values=np.asarray([1], dtype=np.uint8),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="standardized profile metadata",
    )
    assert target.valid_count == 1

    with pytest.raises(ValueError, match="must be binary"):
        TargetValue(
            entity_id="profile-b",
            definition=definition,
            values=np.asarray([2], dtype=np.uint8),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="standardized profile metadata",
        )


def test_geometric_los_contract_checks_name_semantics_and_scalar_domain() -> None:
    definition = _definition(
        "umcd.rt-trajectory.geometric-los.v1",
        semantic="independent Mitsuba segment-intersection geometry oracle",
        units="boolean",
    )
    validate_canonical_target_contract(np.asarray(True), definition, name="geometric_los")
    with pytest.raises(ValueError, match="only valid for target"):
        validate_canonical_target_contract(np.asarray(True), definition, name="los")
    with pytest.raises(ValueError, match="exactly one value"):
        validate_canonical_target_contract(
            np.asarray([True, False]), definition, name="geometric_los"
        )


def test_canonical_cfr_contract_requires_complex_coefficients() -> None:
    definition = _definition(
        "umcd.tdl-pusch.effective-cfr.v1",
        semantic="perfect synchronized effective-subcarrier CFR",
        units="complex channel gain (dimensionless)",
    )
    validate_canonical_target_contract(
        np.asarray([1 + 2j], dtype=np.complex64), definition, name="channel_cfr"
    )
    with pytest.raises(TypeError, match="must use a complex dtype"):
        validate_canonical_target_contract(
            np.asarray([1.0], dtype=np.float32), definition, name="channel_cfr"
        )


def test_target_value_accepts_versioned_complex_cfr_with_masked_zeros() -> None:
    definition = _definition(
        "umcd.tdl-pusch.effective-cfr.v1",
        semantic="perfect synchronized effective-subcarrier CFR",
        units="complex channel gain (dimensionless)",
    )
    target = TargetValue(
        entity_id="episode-link",
        definition=definition,
        values=np.asarray([1.25 - 0.5j, 0.0 + 0.0j], dtype=np.complex64),
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="synthetic_fixture",
    )

    assert target.values.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(
        target.values, np.asarray([1.25 - 0.5j, 0.0 + 0.0j], dtype=np.complex64)
    )

    with pytest.raises(ValueError, match="finite real or complex numeric"):
        TargetValue(
            entity_id="episode-link",
            definition=definition,
            values=np.asarray([complex(float("inf"), 0.0)], dtype=np.complex128),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="synthetic_fixture",
        )


@pytest.mark.parametrize("entity_id", [" episode-link", "episode-link "])
def test_target_value_rejects_padded_entity_identity(entity_id: str) -> None:
    definition = _definition(
        "umcd.tdl-pusch.effective-cfr.v1",
        semantic="perfect synchronized effective-subcarrier CFR",
        units="complex channel gain (dimensionless)",
    )
    with pytest.raises(ValueError, match="canonical string"):
        TargetValue(
            entity_id=entity_id,
            definition=definition,
            values=np.asarray([1.0 + 0.0j], dtype=np.complex64),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="synthetic_fixture",
        )


def test_canonical_target_rejects_padded_definition_fields() -> None:
    definition = _definition(
        "umcd.tdl-pusch.effective-cfr.v1",
        semantic="perfect synchronized effective-subcarrier CFR ",
        units="complex channel gain (dimensionless)",
    )
    with pytest.raises(ValueError, match="semantic must not have surrounding whitespace"):
        TargetValue(
            entity_id="episode-link",
            definition=definition,
            values=np.asarray([1.0 + 0.0j], dtype=np.complex64),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="synthetic_fixture",
        )


@pytest.mark.parametrize("confidence", [" synthetic_fixture", "synthetic_fixture "])
def test_target_value_rejects_padded_confidence_label(confidence: str) -> None:
    definition = _definition(
        "umcd.tdl-pusch.effective-cfr.v1",
        semantic="perfect synchronized effective-subcarrier CFR",
        units="complex channel gain (dimensionless)",
    )
    with pytest.raises(ValueError, match="canonical string"):
        TargetValue(
            entity_id="episode-link",
            definition=definition,
            values=np.asarray([1.0 + 0.0j], dtype=np.complex64),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence=confidence,
        )


def test_canonical_contract_rejects_definition_semantic_drift() -> None:
    definition = _definition(
        "umcd.standardized.profile-los-proxy.v1",
        semantic="geometric visibility",
        units="binary_class",
    )
    with pytest.raises(ValueError, match="semantic or units differ"):
        validate_canonical_target_contract(np.asarray([0]), definition)


@pytest.mark.parametrize("field", ["source", "reference", "aggregation_rule"])
def test_canonical_geometric_target_rejects_provenance_drift(field: str) -> None:
    definition = _definition(
        "umcd.rt-trajectory.geometric-los.v1",
        semantic="independent Mitsuba segment-intersection geometry oracle",
        units="boolean",
    )
    altered = TargetDefinition.versioned(
        semantic=definition.semantic,
        units=definition.units,
        source=("unverified_source" if field == "source" else definition.source),
        reference=("unverified reference" if field == "reference" else definition.reference),
        aggregation_rule=(
            "average labels" if field == "aggregation_rule" else definition.aggregation_rule
        ),
        definition_version=definition.definition_version,
    )
    with pytest.raises(ValueError, match="source|reference|aggregation"):
        validate_canonical_target_contract(np.asarray([True]), altered, name="geometric_los")


def test_nr_storage_admission_enforces_canonical_cfr_value_domain() -> None:
    estimate = NRChannelEstimate(
        h_hat=np.zeros((1, 1, 1, 1, 1, 1, 1), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 1, 1), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 1, 1), dtype=bool),
        metadata={"receiver": "target-contract-fixture"},
    )
    observation = NRObservation(
        sample_id="target-contract-fixture",
        received_grid=np.zeros((1, 1, 1, 1, 1), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(0,),
        receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=15),
    )
    record = NRChannelRecord(
        observation=observation,
        targets={"channel_cfr": np.ones((1,), dtype=np.float32)},
        target_masks={"channel_cfr": np.ones((1,), dtype=bool)},
        provenance={"source": "contract fixture"},
        target_definitions={
            "channel_cfr": _definition(
                "umcd.tdl-pusch.effective-cfr.v1",
                semantic="perfect synchronized effective-subcarrier CFR",
                units="complex channel gain (dimensionless)",
            )
        },
    )
    with pytest.raises(TypeError, match="must use a complex dtype"):
        record.require_versioned_target_definitions()


@pytest.mark.parametrize(
    ("version", "semantic", "target_name", "expected_shape", "wrong_shape"),
    [
        (
            "umcd.standardized.effective-cfr.v1",
            "perfect_synchronized_effective_subcarrier_cfr",
            "channel_cfr",
            (1, 1, 1, 1, 1, 3, 2),
            (1, 1, 1, 1, 1, 3, 1),
        ),
        (
            "umcd.tdl-pusch.effective-cfr.v1",
            "perfect synchronized effective-subcarrier CFR",
            "channel_cfr",
            (1, 1, 1, 1, 1, 3, 2),
            (1, 1, 1, 1, 1, 3, 1),
        ),
        (
            "umcd.rt-trajectory.absolute-cfr.v1",
            "absolute-delay full-path complex CFR on effective PUSCH subcarriers",
            "channel_cfr",
            (2,),
            (1,),
        ),
        (
            "umcd.rt-trajectory.symbol-cfr.v1",
            "absolute-delay full-path CFR with path-Doppler phase evolution sampled at PUSCH OFDM symbols",
            "channel_cfr_ofdm_symbols",
            (3, 2),
            (3, 1),
        ),
    ],
)
def test_nr_cfr_target_shape_matches_its_observation_contract(
    version, semantic, target_name, expected_shape, wrong_shape
) -> None:
    h_hat = np.zeros((1, 1, 1, 1, 1, 3, 2), dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=h_hat,
        error_variance=np.zeros_like(h_hat.real),
        pilot_mask=np.ones((1, 1, 3, 2), dtype=bool),
        metadata={"receiver": "cfr-target-shape-fixture"},
    )
    observation = NRObservation(
        sample_id="cfr-target-shape-fixture",
        received_grid=np.zeros((1, 1, 1, 3, 4), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(1, 2),
        receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=15),
    )
    definition = _definition(
        version,
        semantic=semantic,
        units="complex channel gain (dimensionless)",
    )

    def record(shape):
        return NRChannelRecord(
            observation=observation,
            targets={target_name: np.ones(shape, dtype=np.complex64)},
            target_masks={target_name: np.ones(shape, dtype=bool)},
            provenance={"source": "cfr target shape fixture"},
            target_definitions={target_name: definition},
        )

    record(expected_shape).require_versioned_target_definitions()
    with pytest.raises(ValueError, match="shape .* expected .* from its observation"):
        record(wrong_shape).require_versioned_target_definitions()

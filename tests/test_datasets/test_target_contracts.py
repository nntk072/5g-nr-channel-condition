import datasets.targets as target_contracts
import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation
from datasets.targets import (
    CensoringInterval,
    TargetDefinition,
    TargetValidity,
    TargetValue,
    validate_canonical_target_contract,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _definition(version: str, *, semantic: str, units: str) -> TargetDefinition:
    provenance = {
        "umcd.standardized.profile-los-proxy.v1": (
            "Sionna standardized TDL/CDL profile model",
            "profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth",
            "one profile-category proxy per generated channel record",
        ),
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
        "umcd.standardized.effective-cfr.v2": (
            "standardized Sionna TDL/CDL CFR adapter",
            "profile-relative excess-delay CFR resampled to this resource grid",
            "retain each effective subcarrier coefficient in receiver FFT-bin order; no antenna, symbol, or power averaging",
        ),
        "umcd.tdl-pusch.effective-cfr.v2": (
            "sionna_phy_ofdmchannel_return_channel",
            "OFDMChannel return_channel with the configured TDL profile, carrier, sample rate, and effective FFT-bin mapping",
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
        "umcd.rt-trajectory.absolute-cfr.v2": (
            "sionna_rt_pathsolver",
            "Sionna RT full solver-valid path set evaluated with absolute propagation delays and configured carrier/subcarrier offsets",
            "coherent sum of all retained solver-valid paths per effective subcarrier and frame; no path-count or power normalization",
        ),
        "umcd.rt-trajectory.symbol-cfr.v2": (
            "sionna_rt_paths_cfr_doppler",
            "absolute-delay RT path channel evaluated at the declared NR OFDM symbol start timestamps",
            "coherent sum of retained solver-valid paths per effective subcarrier and OFDM symbol; no path or symbol averaging",
        ),
        "umcd.rms_delay_spread.v1": (
            "path_oracle",
            "absolute_physical_channel_path_oracle",
            "RMS over valid paths after power normalization",
        ),
        "umcd.rt-trajectory.geometric-los.v1": (
            "mitsuba_segment_intersection_oracle",
            "independent finite-segment visibility query for the recorded TX/RX poses and scene geometry",
            "one independent direct-path visibility value per frame",
        ),
        "umcd.f04_rt_dynamic.direct_visibility.v3": (
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment intersection and depth-zero Sionna PathSolver at each recorded pose",
            "one visibility value per trajectory frame; no spatial or temporal pooling",
        ),
        "umcd.f04_rt_dynamic.direct_visibility.v4": (
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment intersection and depth-zero Sionna PathSolver at each recorded pose",
            "one visibility value per trajectory frame; no spatial or temporal pooling",
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
        value_axes={
            "umcd.f04_rt_dynamic.direct_visibility.v4": ("frame",),
            "umcd.rt-trajectory.absolute-cfr.v2": ("effective_subcarrier",),
            "umcd.rt-trajectory.symbol-cfr.v2": ("ofdm_symbol", "effective_subcarrier"),
            "umcd.standardized.effective-cfr.v2": (
                "batch",
                "rx_antenna",
                "rx_stream",
                "ofdm_symbol",
                "tx_stream",
                "tx_antenna",
                "effective_subcarrier",
            ),
            "umcd.tdl-pusch.effective-cfr.v2": (
                "batch",
                "rx_antenna",
                "rx_stream",
                "ofdm_symbol",
                "tx_stream",
                "tx_antenna",
                "effective_subcarrier",
            ),
        }.get(version),
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

    with pytest.raises(ValueError, match=r"fixed \(1,\) storage shape"):
        TargetValue(
            entity_id="profile-c",
            definition=definition,
            values=np.asarray([[1]], dtype=np.uint8),
            valid_mask=np.asarray([[True]]),
            reason_codes=(TargetValidity.VALID,),
            confidence="standardized profile metadata",
        )

    axisful_definition = TargetDefinition.versioned(
        semantic=definition.semantic,
        units=definition.units,
        source=definition.source,
        reference=definition.reference,
        aggregation_rule=definition.aggregation_rule,
        definition_version=definition.definition_version,
        value_axes=("category",),
    )
    with pytest.raises(ValueError, match="must not declare value axes"):
        TargetValue(
            entity_id="profile-d",
            definition=axisful_definition,
            values=np.asarray([1], dtype=np.uint8),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="standardized profile metadata",
        )


def test_current_standardized_pusch_profile_proxy_v1_has_fixed_singleton_shape() -> None:
    definition = _definition(
        "umcd.standardized.profile-los-proxy.v1",
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
    )
    valid = TargetValue(
        entity_id="current-profile",
        definition=definition,
        values=np.asarray([1], dtype=np.uint8),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="standardized profile metadata",
    )
    assert valid.values.shape == (1,)

    with pytest.raises(ValueError, match=r"fixed \(1,\) storage shape"):
        TargetValue(
            entity_id="current-profile-wrong-shape",
            definition=definition,
            values=np.asarray([[1]], dtype=np.uint8),
            valid_mask=np.asarray([[True]]),
            reason_codes=(TargetValidity.VALID,),
            confidence="standardized profile metadata",
        )

    axisful = TargetDefinition.versioned(
        semantic=definition.semantic,
        units=definition.units,
        source=definition.source,
        reference=definition.reference,
        aggregation_rule=definition.aggregation_rule,
        definition_version=definition.definition_version,
        value_axes=("category",),
    )
    with pytest.raises(ValueError, match="must not declare value axes"):
        validate_canonical_target_contract(np.asarray([1]), axisful, name="profile_los_proxy")


def test_historical_static_visibility_v1_keeps_legacy_rank_readable() -> None:
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="independent_segment_visibility_oracle",
        reference="finite scene-segment visibility cross-checked by depth-zero PathSolver",
        aggregation_rule="one direct-path visibility label per link; no temporal or port pooling",
        definition_version="umcd.f03_rt_static.direct_visibility.v1",
    )
    scalar = TargetValue(
        entity_id="link",
        definition=definition,
        values=np.asarray([1], dtype=np.uint8),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="legacy static visibility record",
    )
    matrix = TargetValue(
        entity_id="link",
        definition=definition,
        values=np.asarray([[1]], dtype=np.uint8),
        valid_mask=np.asarray([[True]]),
        reason_codes=(TargetValidity.VALID,),
        confidence="legacy static visibility record",
    )
    assert scalar.values.shape == (1,)
    assert matrix.values.shape == (1, 1)


def test_geometric_los_contract_checks_name_semantics_and_scalar_domain() -> None:
    definition = _definition(
        "umcd.rt-trajectory.geometric-los.v1",
        semantic="independent Mitsuba segment-intersection geometry oracle",
        units="boolean",
    )
    validate_canonical_target_contract(np.asarray([True, False]), definition, name="geometric_los")
    with pytest.raises(ValueError, match="only valid for target"):
        validate_canonical_target_contract(np.asarray(True), definition, name="los")
    with pytest.raises(ValueError, match="values must be binary"):
        validate_canonical_target_contract(np.asarray([True, 2]), definition, name="geometric_los")


def test_legacy_scalar_target_rank_remains_readable_without_axis_inference() -> None:
    definition = _definition(
        "umcd.rt-trajectory.geometric-los.v1",
        semantic="independent Mitsuba segment-intersection geometry oracle",
        units="boolean",
    )
    target = TargetValue(
        entity_id="legacy-link",
        definition=definition,
        values=np.asarray([[1]], dtype=np.uint8),
        valid_mask=np.asarray([[True]]),
        reason_codes=(TargetValidity.VALID,),
        confidence="legacy shape preserved; axes not inferred",
    )
    assert target.values.shape == (1, 1)


def test_frame_visibility_contract_accepts_binary_values_per_frame() -> None:
    definition = _definition(
        "umcd.f04_rt_dynamic.direct_visibility.v3",
        semantic="geometric direct-path visibility",
        units="boolean",
    )
    validate_canonical_target_contract(
        np.asarray([1, 0, 1], dtype=np.uint8), definition, name="geometric_los"
    )
    with pytest.raises(ValueError, match="must be binary"):
        validate_canonical_target_contract(
            np.asarray([1, 2, 0], dtype=np.uint8), definition, name="geometric_los"
        )


def test_dynamic_geometric_los_v4_binds_frame_axis() -> None:
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment "
            "intersection and depth-zero Sionna PathSolver at each recorded pose"
        ),
        aggregation_rule="one visibility value per trajectory frame; no spatial or temporal pooling",
        definition_version="umcd.f04_rt_dynamic.direct_visibility.v4",
        value_axes=("frame",),
    )
    validate_canonical_target_contract(
        np.asarray([1, 0, 1], dtype=np.uint8), definition, name="geometric_los"
    )
    axisless = TargetDefinition.versioned(
        semantic=definition.semantic,
        units=definition.units,
        source=definition.source,
        reference=definition.reference,
        aggregation_rule=definition.aggregation_rule,
        definition_version=definition.definition_version,
    )
    with pytest.raises(ValueError, match=r"explicit \('frame',\)"):
        validate_canonical_target_contract(
            np.asarray([1, 0, 1], dtype=np.uint8), axisless, name="geometric_los"
        )


@pytest.mark.parametrize(
    ("target_name", "version", "semantic", "units", "source", "reference", "aggregation"),
    [
        (
            "profile_los_proxy",
            "umcd.standardized.profile-los-proxy.v1",
            "standardized_profile_los_category_proxy",
            "binary_class",
            "Sionna standardized TDL/CDL profile model",
            "profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth",
            "one profile-category proxy per generated channel record",
        ),
        (
            "geometric_los",
            "umcd.f03_rt_static.direct_visibility.v2",
            "geometric direct-path visibility",
            "boolean",
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle visibility cross-checked by Mitsuba "
            "segment intersection and depth-zero PathSolver",
            "one direct-path visibility label per link; no temporal or port pooling",
        ),
        (
            "geometric_los",
            "umcd.rt-trajectory.geometric-los.v2",
            "digest-bound NumPy loaded-scene direct-visibility oracle",
            "boolean",
            "numpy_loaded_scene_triangle_inventory",
            "loaded-scene triangle query cross-checked with Mitsuba segment intersection "
            "and depth-zero Sionna PathSolver for each recorded TX/RX pose and geometry digest",
            "one independent direct-path visibility value per frame",
        ),
        (
            "geometric_los",
            "umcd.f04_rt_dynamic.direct_visibility.v3",
            "geometric direct-path visibility",
            "boolean",
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment "
            "intersection and depth-zero Sionna PathSolver at each recorded pose",
            "one visibility value per trajectory frame; no spatial or temporal pooling",
        ),
        (
            "geometric_los",
            "umcd.f04_rt_dynamic.direct_visibility.v4",
            "geometric direct-path visibility",
            "boolean",
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment "
            "intersection and depth-zero Sionna PathSolver at each recorded pose",
            "one visibility value per trajectory frame; no spatial or temporal pooling",
        ),
    ],
)
def test_current_producer_target_versions_pin_semantics_and_provenance(
    target_name, version, semantic, units, source, reference, aggregation
) -> None:
    definition = TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=source,
        reference=reference,
        aggregation_rule=aggregation,
        definition_version=version,
        value_axes=("frame",) if version == "umcd.f04_rt_dynamic.direct_visibility.v4" else None,
    )
    validate_canonical_target_contract(np.asarray([True]), definition, name=target_name)

    altered = TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=f"{source} altered",
        reference=reference,
        aggregation_rule=aggregation,
        definition_version=version,
    )
    with pytest.raises(ValueError, match="source differs"):
        validate_canonical_target_contract(np.asarray([True]), altered, name=target_name)


def test_canonical_target_admission_fails_closed_without_provenance_pin(monkeypatch) -> None:
    definition = _definition(
        "umcd.rt-trajectory.geometric-los.v1",
        semantic="independent Mitsuba segment-intersection geometry oracle",
        units="boolean",
    )
    monkeypatch.delitem(
        target_contracts._CANONICAL_TARGET_PROVENANCE,
        definition.definition_version,
    )

    with pytest.raises(RuntimeError, match="has no provenance pin"):
        validate_canonical_target_contract(np.asarray([True]), definition, name="geometric_los")


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


def test_rms_delay_spread_contract_requires_nonnegative_real_seconds() -> None:
    definition = _definition(
        "umcd.rms_delay_spread.v1",
        semantic="rms_delay_spread",
        units="s",
    )
    validate_canonical_target_contract(
        np.asarray([0.0, 25e-9]), definition, name="rms_delay_spread"
    )
    with pytest.raises(ValueError, match="must be nonnegative"):
        validate_canonical_target_contract(
            np.asarray([25e-9, -1e-9]), definition, name="rms_delay_spread"
        )
    with pytest.raises(TypeError, match="real numeric dtype"):
        validate_canonical_target_contract(
            np.asarray([25e-9 + 0j]), definition, name="rms_delay_spread"
        )
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="must be finite"):
            validate_canonical_target_contract(
                np.asarray([invalid]), definition, name="rms_delay_spread"
            )


def test_canonical_complex_cfr_contract_rejects_nonfinite_values() -> None:
    definition = _definition(
        "umcd.standardized.effective-cfr.v1",
        semantic="perfect_synchronized_effective_subcarrier_cfr",
        units="complex channel gain (dimensionless)",
    )

    for invalid in (complex(np.nan, 0.0), complex(0.0, np.inf)):
        with pytest.raises(ValueError, match="must contain finite values"):
            validate_canonical_target_contract(
                np.asarray([invalid], dtype=np.complex128), definition, name="channel_cfr"
            )


def test_rms_delay_spread_v2_binds_frame_axis_in_its_digest() -> None:
    definition = TargetDefinition.versioned(
        semantic="rms_delay_spread",
        units="s",
        source="path_oracle",
        reference="absolute_physical_channel_path_oracle",
        aggregation_rule=(
            "RMS over valid per-path powers in the declared frame and delay reference"
        ),
        definition_version="umcd.rms_delay_spread.v2",
        value_axes=("frame",),
    )
    target = TargetValue(
        entity_id="link-1",
        definition=definition,
        values=np.asarray([0.0, 20e-9]),
        valid_mask=np.ones(2, dtype=bool),
        reason_codes=(TargetValidity.VALID, TargetValidity.VALID),
        confidence="synthetic_contract_check",
    )
    assert definition.value_axes == ("frame",)
    assert target.values.shape == (2,)

    with pytest.raises(ValueError, match="rank must match its declared value axes"):
        TargetValue(
            entity_id="link-1",
            definition=definition,
            values=np.asarray([[0.0, 20e-9]]),
            valid_mask=np.ones((1, 2), dtype=bool),
            reason_codes=(TargetValidity.VALID, TargetValidity.VALID),
            confidence="synthetic_contract_check",
        )
    with pytest.raises(TypeError, match="tuple or list"):
        TargetDefinition.versioned(
            semantic="rms_delay_spread",
            units="s",
            source="path_oracle",
            reference="absolute_physical_channel_path_oracle",
            aggregation_rule=(
                "RMS over valid per-path powers in the declared frame and delay reference"
            ),
            definition_version="umcd.rms_delay_spread.v2",
            value_axes="frame",  # type: ignore[arg-type]
        )

    missing_axis = TargetDefinition.versioned(
        semantic="rms_delay_spread",
        units="s",
        source="path_oracle",
        reference="absolute_physical_channel_path_oracle",
        aggregation_rule=(
            "RMS over valid per-path powers in the declared frame and delay reference"
        ),
        definition_version="umcd.rms_delay_spread.v2",
    )
    with pytest.raises(ValueError, match="v2 needs explicit"):
        TargetValue(
            entity_id="link-1",
            definition=missing_axis,
            values=np.asarray([20e-9]),
            valid_mask=np.ones(1, dtype=bool),
            reason_codes=(TargetValidity.VALID,),
            confidence="synthetic_contract_check",
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


def test_censored_target_requires_finite_nonempty_interval_bounds():
    definition = _definition(
        "umcd.synthetic.censored-delay.v1",
        semantic="censored delay",
        units="s",
    )
    target = TargetValue(
        entity_id="episode-link",
        definition=definition,
        values=np.asarray([0.0]),
        valid_mask=np.asarray([False]),
        reason_codes=(TargetValidity.CENSORED,),
        confidence="synthetic_fixture",
        censoring_intervals=(CensoringInterval(lower=10e-9),),
    )
    assert target.censoring_intervals == (CensoringInterval(lower=10e-9),)

    with pytest.raises(ValueError, match="require an interval"):
        TargetValue(
            entity_id="episode-link",
            definition=definition,
            values=np.asarray([0.0]),
            valid_mask=np.asarray([False]),
            reason_codes=(TargetValidity.CENSORED,),
            confidence="synthetic_fixture",
        )
    with pytest.raises(ValueError, match="must not carry one"):
        TargetValue(
            entity_id="episode-link",
            definition=definition,
            values=np.asarray([0.0]),
            valid_mask=np.asarray([False]),
            reason_codes=(TargetValidity.NOT_OBSERVABLE,),
            confidence="synthetic_fixture",
            censoring_intervals=(CensoringInterval(upper=10e-9),),
        )


@pytest.mark.parametrize(
    "bounds",
    [
        {"lower": None, "upper": None},
        {"lower": 2.0, "upper": 1.0},
        {"lower": float("nan"), "upper": None},
        {"lower": True, "upper": None},
    ],
)
def test_censoring_interval_rejects_invalid_numeric_bounds(bounds):
    with pytest.raises(ValueError):
        CensoringInterval(**bounds)


def test_censoring_interval_normalization_overflow_is_a_domain_error():
    with pytest.raises(ValueError, match="finite real number or null"):
        CensoringInterval(lower=10**10000)


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


def test_binary_target_contract_checks_value_dtype_and_mask_contract():
    definition = _definition(
        "umcd.standardized.profile-los-proxy.v1",
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
    )
    with pytest.raises(ValueError, match="exactly one value"):
        validate_canonical_target_contract(
            np.asarray([True, False]), definition, name="profile_los_proxy"
        )
    with pytest.raises(TypeError, match="real boolean or numeric dtype"):
        validate_canonical_target_contract(
            np.asarray(["true"]), definition, name="profile_los_proxy"
        )
    with pytest.raises(TypeError, match="mask.*must be boolean"):
        validate_canonical_target_contract(
            np.asarray([True]),
            definition,
            name="profile_los_proxy",
            valid_mask=np.asarray([1]),
        )
    with pytest.raises(ValueError, match="mask.*wrong shape"):
        validate_canonical_target_contract(
            np.asarray([True]),
            definition,
            name="profile_los_proxy",
            valid_mask=np.asarray([[True]]),
        )


def test_binary_target_contract_converts_tensor_values_and_masks():
    torch = pytest.importorskip("torch")
    definition = _definition(
        "umcd.standardized.profile-los-proxy.v1",
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
    )
    validate_canonical_target_contract(
        torch.tensor(True),
        definition,
        name="profile_los_proxy",
        valid_mask=torch.tensor(True),
    )


@pytest.mark.parametrize(
    ("axes", "error", "message"),
    [
        ("frame", TypeError, "tuple or list"),
        (("frame", " frame"), ValueError, "canonical strings"),
        (("frame", "frame"), ValueError, "must be unique"),
    ],
)
def test_target_definition_rejects_invalid_value_axes(axes, error, message):
    with pytest.raises(error, match=message):
        TargetDefinition("semantic", "unit", "a" * 64, value_axes=axes)


def _simple_definition() -> TargetDefinition:
    return TargetDefinition.versioned(
        semantic="fixture target",
        units="unit",
        source="fixture producer",
        reference="fixture physical reference",
        aggregation_rule="one value per fixture entity",
        definition_version="tests.target.v1",
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"definition": object()}, TypeError, "definition must be a TargetDefinition"),
        ({"valid_mask": np.asarray([1])}, TypeError, "valid_mask must use boolean dtype"),
        ({"valid_mask": np.asarray([True, False])}, ValueError, "exactly the target value shape"),
        ({"reason_codes": ()}, ValueError, "one TargetValidity per target element"),
        ({"censoring_intervals": ()}, ValueError, "one interval or null per target element"),
        ({"valid_mask": np.asarray([False])}, ValueError, "valid_mask and reason_codes disagree"),
        (
            {
                "values": np.asarray([0j]),
                "valid_mask": np.asarray([False]),
                "reason_codes": (TargetValidity.CENSORED,),
                "censoring_intervals": (CensoringInterval(lower=1.0),),
            },
            ValueError,
            "cannot describe complex target values",
        ),
    ],
)
def test_target_value_rejects_malformed_definition_and_element_metadata(overrides, error, message):
    arguments = {
        "entity_id": "link-1",
        "definition": _simple_definition(),
        "values": np.asarray([1.0]),
        "valid_mask": np.asarray([True]),
        "reason_codes": (TargetValidity.VALID,),
        "confidence": "fixture confidence",
    }
    arguments.update(overrides)

    with pytest.raises(error, match=message):
        TargetValue(**arguments)


def test_target_value_rejects_empty_values_and_nonfinite_censoring_interval():
    definition = _simple_definition()
    with pytest.raises(ValueError, match="at least one element"):
        TargetValue(
            entity_id="link-1",
            definition=definition,
            values=np.asarray([]),
            valid_mask=np.asarray([], dtype=bool),
            reason_codes=(),
            confidence="fixture confidence",
        )


def test_target_definition_digest_requires_canonical_nonempty_fields_and_axes():
    kwargs = {
        "semantic": "semantic",
        "units": "unit",
        "reference": "physical reference",
        "aggregation_rule": "one value",
        "source": "source",
        "definition_version": "test.v1",
    }
    with pytest.raises(ValueError, match="fields must be non-empty strings"):
        target_contracts.target_definition_digest(**{**kwargs, "units": " "})
    with pytest.raises(TypeError, match="axes must be a tuple or list"):
        target_contracts.target_definition_digest(**kwargs, value_axes="frame")
    with pytest.raises(ValueError, match="axes cannot be empty"):
        target_contracts.target_definition_digest(**kwargs, value_axes=())
    with pytest.raises(ValueError, match="canonical strings"):
        target_contracts.target_definition_digest(**kwargs, value_axes=(" frame",))
    with pytest.raises(ValueError, match="must be unique"):
        target_contracts.target_definition_digest(**kwargs, value_axes=("frame", "frame"))


def test_legacy_target_migration_requires_semantic_units_and_source():
    with pytest.raises(ValueError, match="require semantic, units, and source strings"):
        target_contracts.migrate_legacy_target_definition({"semantic": "legacy"})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"reference": "unspecified_legacy"}, "explicit physical reference"),
        ({"aggregation_rule": "unspecified_legacy"}, "explicit aggregation rule"),
        ({"definition_version": "unversioned_legacy"}, "explicit version"),
    ],
)
def test_target_definition_requires_reference_aggregation_and_version(fields, message):
    arguments = {
        "semantic": "fixture target",
        "units": "unit",
        "source": "fixture producer",
        "reference": "fixture physical reference",
        "aggregation_rule": "one value per fixture entity",
        "definition_version": "tests.target.v1",
    }
    arguments.update(fields)
    definition = TargetDefinition.versioned(**arguments)

    with pytest.raises(ValueError, match=message):
        definition.require_versioned()


def test_unavailable_reason_counts_and_definition_digest_input_validation():
    target = TargetValue(
        entity_id="link-1",
        definition=_simple_definition(),
        values=np.asarray([1.0, 0.0]),
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="fixture confidence",
    )
    assert target.unavailable_counts == {TargetValidity.NOT_OBSERVABLE: 1}
    assert len(target_contracts.definition_digest("canonical target text")) == 64

    with pytest.raises(ValueError, match="canonical_definition must be a non-empty string"):
        target_contracts.definition_digest(" ")
    with pytest.raises(ValueError, match="canonical_definition must be a non-empty string"):
        target_contracts.definition_digest(None)


@pytest.mark.parametrize("reference", [None, "", "unspecified_legacy"])
def test_legacy_target_migration_marks_unrecorded_physical_reference(reference):
    payload = {"semantic": "legacy", "units": "unit", "source": "source"}
    if reference is not None:
        payload["reference"] = reference

    migrated = target_contracts.migrate_legacy_target_definition(payload)

    assert migrated.reference == "not recorded by legacy target schema v1"


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
            "umcd.standardized.effective-cfr.v2",
            "perfect_synchronized_effective_subcarrier_cfr",
            "channel_cfr",
            (1, 1, 1, 1, 1, 3, 2),
            (1, 1, 1, 1, 1, 3, 1),
        ),
        (
            "umcd.tdl-pusch.effective-cfr.v2",
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
            "umcd.rt-trajectory.absolute-cfr.v2",
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
        (
            "umcd.rt-trajectory.symbol-cfr.v2",
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


def test_nr_trajectory_geometric_visibility_is_one_scalar_per_record() -> None:
    h_hat = np.zeros((1, 1, 1, 1, 1, 3, 2), dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=h_hat,
        error_variance=np.zeros_like(h_hat.real),
        pilot_mask=np.ones((1, 1, 3, 2), dtype=bool),
        metadata={"receiver": "trajectory-geometric-los-shape-fixture"},
    )
    observation = NRObservation(
        sample_id="trajectory-geometric-los-shape-fixture",
        received_grid=np.zeros((1, 1, 1, 3, 4), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(1, 2),
        receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=15),
    )
    definition = TargetDefinition.versioned(
        semantic="digest-bound NumPy loaded-scene direct-visibility oracle",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "loaded-scene triangle query cross-checked with Mitsuba segment intersection "
            "and depth-zero Sionna PathSolver for each recorded TX/RX pose and geometry digest"
        ),
        aggregation_rule="one independent direct-path visibility value per frame",
        definition_version="umcd.rt-trajectory.geometric-los.v2",
    )

    def record(value: np.ndarray) -> NRChannelRecord:
        return NRChannelRecord(
            observation=observation,
            targets={"geometric_los": value},
            target_masks={"geometric_los": np.ones(value.shape, dtype=bool)},
            provenance={"source": "trajectory geometric LOS shape fixture"},
            target_definitions={"geometric_los": definition},
        )

    record(np.asarray(True)).require_versioned_target_definitions()
    with pytest.raises(ValueError, match=r"shape .* expected \(\) from its observation"):
        record(np.asarray([True])).require_versioned_target_definitions()


def test_nr_standardized_profile_proxy_uses_fixed_singleton_storage_shape() -> None:
    h_hat = np.zeros((1, 1, 1, 1, 1, 3, 2), dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=h_hat,
        error_variance=np.zeros_like(h_hat.real),
        pilot_mask=np.ones((1, 1, 3, 2), dtype=bool),
        metadata={"receiver": "profile-proxy-shape-fixture"},
    )
    observation = NRObservation(
        sample_id="profile-proxy-shape-fixture",
        received_grid=np.zeros((1, 1, 1, 3, 4), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(1, 2),
        receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=15),
    )
    definition = TargetDefinition.versioned(
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
        source="Sionna standardized TDL/CDL profile model",
        reference=(
            "profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth"
        ),
        aggregation_rule="one profile-category proxy per generated channel record",
        definition_version="umcd.standardized.profile-los-proxy.v1",
    )

    def record(value: np.ndarray) -> NRChannelRecord:
        return NRChannelRecord(
            observation=observation,
            targets={"profile_los_proxy": value},
            target_masks={"profile_los_proxy": np.ones(value.shape, dtype=bool)},
            provenance={"source": "profile proxy shape fixture"},
            target_definitions={"profile_los_proxy": definition},
        )

    record(np.asarray([1], dtype=np.uint8)).require_versioned_target_definitions()
    with pytest.raises(ValueError, match=r"shape .* expected \(1,\) from its observation"):
        record(np.asarray([[1]], dtype=np.uint8)).require_versioned_target_definitions()


@pytest.mark.parametrize(
    ("version", "semantic", "units", "expected_axes"),
    [
        (
            "umcd.rt-trajectory.absolute-cfr.v2",
            "absolute-delay full-path complex CFR on effective PUSCH subcarriers",
            "complex channel gain (dimensionless)",
            ("effective_subcarrier",),
        ),
        (
            "umcd.rt-trajectory.symbol-cfr.v2",
            "absolute-delay full-path CFR with path-Doppler phase evolution sampled at PUSCH OFDM symbols",
            "complex channel gain (dimensionless)",
            ("ofdm_symbol", "effective_subcarrier"),
        ),
    ],
)
def test_rt_cfr_v2_requires_semantic_value_axes(version, semantic, units, expected_axes):
    definition = _definition(version, semantic=semantic, units=units)
    assert definition.value_axes == expected_axes
    validate_canonical_target_contract(
        np.ones((2,) if len(expected_axes) == 1 else (3, 2), dtype=np.complex64),
        definition,
        name="channel_cfr" if len(expected_axes) == 1 else "channel_cfr_ofdm_symbols",
    )
    axisless = TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=definition.source,
        reference=definition.reference,
        aggregation_rule=definition.aggregation_rule,
        definition_version=version,
    )
    with pytest.raises(ValueError, match="explicit .* value axes"):
        validate_canonical_target_contract(
            np.ones((2,) if len(expected_axes) == 1 else (3, 2), dtype=np.complex64),
            axisless,
        )

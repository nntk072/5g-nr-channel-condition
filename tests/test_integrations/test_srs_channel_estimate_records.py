import numpy as np
import pytest
from integrations.sionna.srs.srs_constants.channel_estimates import (
    NRSRSMultiportChannelEstimate,
    NRSRSSinglePortChannelEstimate,
)


def single_port_record(**overrides):
    values = {
        "channel_by_re": np.array([[1 + 2j, 0], [0, 3 - 1j]]),
        "mask": np.array([[True, False], [False, True]]),
        "symbol_indices": (0, 1),
    }
    values.update(overrides)
    return NRSRSSinglePortChannelEstimate(**values)


def test_single_port_record_freezes_valid_samples_and_preserves_symbol_order():
    original = np.array([[1 + 2j, 0], [0, 3 - 1j]])
    record = single_port_record(channel_by_re=original, symbol_indices=(1, 0))

    assert record.symbol_indices == (1, 0)
    assert not record.channel_by_re.flags.writeable
    assert not record.mask.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        record.channel_by_re[0, 0] = 9j


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("channel_by_re", np.ones(2, dtype=np.complex128), ValueError, "must be complex"),
        ("channel_by_re", np.ones((2, 2)), ValueError, "must be complex"),
        ("mask", np.ones((2, 1), dtype=np.bool_), ValueError, "must be boolean"),
        ("mask", np.ones((2, 2), dtype=np.uint8), ValueError, "must be boolean"),
        (
            "channel_by_re",
            np.array([[np.inf + 0j, 0], [0, 1j]]),
            ValueError,
            "finite values",
        ),
        (
            "channel_by_re",
            np.array([[1j, 1], [0, 1j]]),
            ValueError,
            "zero outside",
        ),
        ("symbol_indices", (), ValueError, "unique valid"),
        ("symbol_indices", (True,), ValueError, "unique valid"),
        ("symbol_indices", (0, 0), ValueError, "unique valid"),
        ("symbol_indices", (2,), ValueError, "unique valid"),
    ],
)
def test_single_port_record_rejects_invalid_field_contract(field, value, error, message):
    with pytest.raises(error, match=message):
        single_port_record(**{field: value})


def multiport_record(**overrides):
    values = {
        "channel_by_rx_port_subcarrier": np.array([[[1 + 0j], [2 + 0j]]]),
        "noise_covariance_by_rx_subcarrier": np.array([[[[1 + 0j, 0], [0, 1 + 0j]]]]),
        "valid_subcarrier_mask": np.array([True]),
        "rank_by_subcarrier": np.array([2]),
        "condition_number_by_subcarrier": (1.0,),
        "reason_codes": (None,),
        "noise_variance_by_rx_ant": np.array([0.5]),
        "symbol_indices": (0, 1),
    }
    values.update(overrides)
    return NRSRSMultiportChannelEstimate(**values)


def test_multiport_record_freezes_arrays_and_accepts_symbol_noise_profile():
    symbol_noise = np.array([[0.25, 0.75]])
    record = multiport_record(noise_variance_by_rx_symbol=symbol_noise)

    assert record.noise_variance_by_rx_ant[0] == 0.5
    assert record.noise_variance_by_rx_symbol is not None
    assert not record.noise_variance_by_rx_symbol.flags.writeable
    assert not record.channel_by_rx_port_subcarrier.flags.writeable
    assert not record.noise_covariance_by_rx_subcarrier.flags.writeable
    assert not record.valid_subcarrier_mask.flags.writeable
    assert not record.rank_by_subcarrier.flags.writeable


def test_multiport_record_accepts_zeroed_unavailable_subcarrier():
    record = multiport_record(
        channel_by_rx_port_subcarrier=np.zeros((1, 2, 1), dtype=np.complex128),
        noise_covariance_by_rx_subcarrier=np.zeros((1, 1, 2, 2), dtype=np.complex128),
        valid_subcarrier_mask=np.array([False]),
        rank_by_subcarrier=np.array([1]),
        condition_number_by_subcarrier=(None,),
        reason_codes=("rank_deficient_port_signature",),
    )

    assert record.reason_codes == ("rank_deficient_port_signature",)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("channel_by_rx_port_subcarrier", np.ones((1, 2, 1)), ValueError, "must be complex"),
        (
            "channel_by_rx_port_subcarrier",
            np.zeros((1, 2, 0), dtype=np.complex128),
            ValueError,
            "dimensions must be nonempty",
        ),
        (
            "noise_covariance_by_rx_subcarrier",
            np.zeros((1, 2, 2, 2), dtype=np.complex128),
            ValueError,
            "noise covariance shape",
        ),
        (
            "noise_covariance_by_rx_subcarrier",
            np.zeros((1, 1, 2, 2)),
            TypeError,
            "must be complex-valued",
        ),
        ("valid_subcarrier_mask", np.array([1]), ValueError, "must be boolean"),
        ("rank_by_subcarrier", np.array([2.0]), ValueError, "must be an integer"),
        ("condition_number_by_subcarrier", (), ValueError, "rosters must match"),
        ("noise_variance_by_rx_ant", np.ones((1, 1)), ValueError, "must be real"),
        ("noise_variance_by_rx_ant", np.array([True]), ValueError, "must be real"),
        ("symbol_indices", (), ValueError, "unique nonnegative"),
        ("symbol_indices", (True,), ValueError, "unique nonnegative"),
        ("symbol_indices", (0, 0), ValueError, "unique nonnegative"),
        ("symbol_indices", (-1,), ValueError, "unique nonnegative"),
        (
            "noise_variance_by_rx_symbol",
            np.ones((2, 2)),
            ValueError,
            "positive real",
        ),
        (
            "noise_variance_by_rx_symbol",
            np.array([[1, np.nan]]),
            ValueError,
            "positive real",
        ),
        (
            "noise_variance_by_rx_symbol",
            np.array([[1, 0]]),
            ValueError,
            "positive real",
        ),
        (
            "channel_by_rx_port_subcarrier",
            np.array([[[np.nan + 0j], [1 + 0j]]]),
            ValueError,
            "contain invalid values",
        ),
        (
            "noise_variance_by_rx_ant",
            np.array([0.0]),
            ValueError,
            "contain invalid values",
        ),
        ("rank_by_subcarrier", np.array([-1]), ValueError, "contain invalid values"),
        ("rank_by_subcarrier", np.array([3]), ValueError, "contain invalid values"),
        (
            "condition_number_by_subcarrier",
            (None,),
            ValueError,
            "full rank and finite conditioning",
        ),
        (
            "condition_number_by_subcarrier",
            (np.inf,),
            ValueError,
            "full rank and finite conditioning",
        ),
        ("reason_codes", ("unexpected",), ValueError, "cannot have an unavailable reason"),
    ],
)
def test_multiport_record_rejects_invalid_field_contract(field, value, error, message):
    with pytest.raises(error, match=message):
        multiport_record(**{field: value})


def test_multiport_record_rejects_invalid_unavailable_reason_and_nonzero_values():
    with pytest.raises(ValueError, match="require a reason code"):
        multiport_record(
            valid_subcarrier_mask=np.array([False]),
            rank_by_subcarrier=np.array([1]),
            condition_number_by_subcarrier=(None,),
            reason_codes=(None,),
        )

    with pytest.raises(ValueError, match="must have zero estimates"):
        multiport_record(
            valid_subcarrier_mask=np.array([False]),
            rank_by_subcarrier=np.array([1]),
            condition_number_by_subcarrier=(None,),
            reason_codes=("rank_deficient",),
        )


def test_multiport_record_rejects_empty_or_nonstring_unavailable_reasons():
    for reason in ("", 3):
        with pytest.raises(ValueError, match="require a reason code"):
            multiport_record(
                valid_subcarrier_mask=np.array([False]),
                rank_by_subcarrier=np.array([1]),
                condition_number_by_subcarrier=(None,),
                reason_codes=(reason,),
            )

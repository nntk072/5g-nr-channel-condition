"""Native checks for the 32-bit BCH-to-864-bit PBCH coding adapter."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
polar_module = pytest.importorskip("sionna.phy.fec.polar")

from integrations.sionna import (  # noqa: E402
    PBCHMIB,
    map_nr_ssb_resource_grid,  # noqa: E402
    nr_pbch_bch_decode,
    nr_pbch_mib_parse,
    nr_pbch_mib_serialize,
    nr_pbch_payload_generate,
    nr_pbch_payload_recover,
    nr_pbch_qpsk_modulate,
    nr_pbch_scramble_bits,
)
from integrations.sionna import nr_pbch_bch_encode as public_nr_pbch_bch_encode  # noqa: E402
from integrations.sionna.pbch_bch import nr_pbch_bch_encode  # noqa: E402


@pytest.mark.native_sionna
def test_pbch_bch_encoding_matches_512_bit_sionna_mother_code_and_repetition() -> None:
    payload = np.arange(32, dtype=np.uint8) % 2
    actual = nr_pbch_bch_encode(payload)
    reference_encoder = polar_module.Polar5GEncoder(
        32,
        512,
        channel_type="downlink",
        device="cpu",
    )
    reference = (
        reference_encoder(torch.as_tensor(payload, dtype=torch.float32).unsqueeze(0))
        .reshape(-1)
        .to(torch.uint8)
        .numpy()
    )

    np.testing.assert_array_equal(actual[:512], reference)
    np.testing.assert_array_equal(actual[512:], reference[:352])
    assert actual.shape == (864,)
    assert set(np.unique(actual).tolist()) <= {0, 1}
    assert not actual.flags.writeable
    assert public_nr_pbch_bch_encode is nr_pbch_bch_encode


@pytest.mark.native_sionna
def test_pbch_bch_decoder_recovers_batched_payload_and_crc() -> None:
    payloads = np.arange(8 * 32, dtype=np.uint8).reshape(8, 32) % 2
    llr_batches = []
    for payload in payloads:
        coded = nr_pbch_bch_encode(payload)
        llr_batches.append(5.0 * (2.0 * coded.astype(np.float64) - 1.0))

    recovered, crc_valid = nr_pbch_bch_decode(np.stack(llr_batches))

    np.testing.assert_array_equal(recovered, payloads)
    np.testing.assert_array_equal(crc_valid, np.ones(8, dtype=np.bool_))
    assert not recovered.flags.writeable
    assert not crc_valid.flags.writeable


@pytest.mark.native_sionna
def test_pbch_bch_decoder_corrects_seeded_llr_noise() -> None:
    rng = np.random.default_rng(20261024)
    payloads = rng.integers(0, 2, size=(8, 32), dtype=np.uint8)
    llr_batches = []
    for payload in payloads:
        coded = nr_pbch_bch_encode(payload)
        clean_llrs = 5.0 * (2.0 * coded.astype(np.float64) - 1.0)
        llr_batches.append(clean_llrs + rng.normal(0.0, 1.5, size=864))

    recovered, crc_valid = nr_pbch_bch_decode(np.stack(llr_batches))

    np.testing.assert_array_equal(recovered, payloads)
    np.testing.assert_array_equal(crc_valid, np.ones(8, dtype=np.bool_))


@pytest.mark.native_sionna
def test_pbch_bch_output_composes_with_scrambler_modulator_and_ssb_grid() -> None:
    transport_block = np.arange(32, dtype=np.uint8) % 2
    coded_bits = nr_pbch_bch_encode(transport_block)
    scrambled_bits = nr_pbch_scramble_bits(
        coded_bits,
        physical_cell_id=317,
        ssb_index_in_half_frame=2,
        max_candidate_blocks_per_half_frame=4,
    )
    symbols = nr_pbch_qpsk_modulate(scrambled_bits)
    grid = map_nr_ssb_resource_grid(
        105,
        2,
        symbols,
        ssb_index_in_half_frame=2,
        half_frame_index=0,
        max_candidate_blocks_per_half_frame=4,
    )

    assert grid.physical_cell_id == 317
    assert grid.grid.shape == (4, 240)
    assert np.count_nonzero(grid.grid) == 830


@pytest.mark.native_sionna
def test_pbch_payload_bch_and_physical_data_chain_round_trips() -> None:
    sfn = 0b1010110101
    mib = PBCHMIB(
        system_frame_number_msb=sfn >> 4,
        subcarrier_spacing_common="scs30or120",
        ssb_subcarrier_offset=9,
        dmrs_type_a_position="pos3",
        control_resource_set_zero=11,
        search_space_zero=6,
        cell_barred="notBarred",
        intra_freq_reselection="allowed",
        spare=0,
    )
    mib_bits = nr_pbch_mib_serialize(mib)
    payload = nr_pbch_payload_generate(
        mib_bits,
        physical_cell_id=317,
        system_frame_number=sfn,
        half_frame_index=1,
        ssb_index_in_half_frame=6,
        max_candidate_blocks_per_half_frame=8,
        k_ssb_msb=1,
        pdcch_repetition=1,
    )
    coded = nr_pbch_bch_encode(payload)
    physical_scrambled = nr_pbch_scramble_bits(
        coded,
        physical_cell_id=317,
        ssb_index_in_half_frame=6,
        max_candidate_blocks_per_half_frame=8,
    )
    qpsk = nr_pbch_qpsk_modulate(physical_scrambled)
    hard_bits = np.empty(864, dtype=np.uint8)
    hard_bits[0::2] = qpsk.real < 0
    hard_bits[1::2] = qpsk.imag < 0
    physically_descrambled = nr_pbch_scramble_bits(
        hard_bits,
        physical_cell_id=317,
        ssb_index_in_half_frame=6,
        max_candidate_blocks_per_half_frame=8,
    )
    llrs = 5.0 * (2.0 * physically_descrambled.astype(np.float64) - 1.0)
    decoded, crc_valid = nr_pbch_bch_decode(llrs)
    recovered = nr_pbch_payload_recover(decoded, 317, max_candidate_blocks_per_half_frame=8)

    np.testing.assert_array_equal(crc_valid, True)
    np.testing.assert_array_equal(recovered.higher_layer_bits, mib_bits)
    assert nr_pbch_mib_parse(recovered.higher_layer_bits) == mib
    assert recovered.system_frame_number == sfn
    assert recovered.half_frame_index == 1
    assert recovered.k_ssb_msb == 1
    assert recovered.pdcch_repetition == 1


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (np.zeros(31, dtype=np.uint8), ValueError, "shape"),
        (np.full(32, 2, dtype=np.uint8), ValueError, "binary"),
        (np.full(32, -1, dtype=np.int8), ValueError, "binary"),
        (np.zeros(32, dtype=np.float64), ValueError, "binary"),
    ],
)
def test_pbch_bch_encoder_rejects_invalid_transport_blocks(
    payload: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        nr_pbch_bch_encode(payload)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("llrs", "error", "message"),
    [
        (np.zeros(863, dtype=np.float32), ValueError, "864"),
        (np.full(864, np.nan, dtype=np.float32), ValueError, "finite"),
        (np.zeros(864, dtype=np.complex64), ValueError, "finite real"),
    ],
)
def test_pbch_bch_decoder_rejects_invalid_llrs(
    llrs: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        nr_pbch_bch_decode(llrs)

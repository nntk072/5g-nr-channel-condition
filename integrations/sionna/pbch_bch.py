"""PBCH BCH CRC, Polar coding, and 864-bit rate matching."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from numpy.typing import NDArray

_BCH_INPUT_BITS = 32
_POLAR_MOTHER_BITS = 512
_RATE_MATCHED_BITS = 864
_REPETITION_BITS = _RATE_MATCHED_BITS - _POLAR_MOTHER_BITS


def nr_pbch_bch_encode(
    scrambled_transport_block: NDArray[np.generic], *, device: str = "cpu"
) -> NDArray[np.uint8]:
    """Encode one 32-bit TS 38.212 BCH input into 864 rate-matched bits.

    The input is the PBCH transport-block bit sequence after payload
    generation/interleaving and the TS 38.212 clause 7.1.2 payload-scrambling
    step. Sionna's CRC24C + downlink Polar encoder supplies CRC attachment,
    input interleaving, N=512 Polar coding, and sub-block interleaving. PBCH
    uses E=864 with no coded-bit interleaver; its rate matching repeats the
    sub-block-interleaved 512-bit mother codeword until 864 bits are emitted.

    This function requires the optional pinned ``sionna==2.2.0`` dependency.
    It does not perform TS 38.211 physical-channel scrambling, QPSK mapping,
    or SSB-grid placement.
    """
    # The payload generator intentionally returns immutable arrays; PyTorch
    # warns that ``as_tensor`` on a non-writable NumPy view has undefined
    # write behavior, so detach an owned writable copy at this boundary.
    payload = np.array(_binary_payload(scrambled_transport_block), copy=True)
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    if not device:
        raise ValueError("device must not be empty")

    encoder = _polar_encoder(device)
    import torch

    payload_tensor = torch.as_tensor(payload, dtype=torch.float32, device=device).unsqueeze(0)
    encoded = encoder(payload_tensor).reshape(-1).to(torch.uint8).detach().cpu().numpy()
    if encoded.shape != (_POLAR_MOTHER_BITS,):
        raise RuntimeError("pinned Sionna Polar encoder returned an unexpected mother-code length")
    if np.any(encoded > 1):
        raise RuntimeError("pinned Sionna Polar encoder returned non-binary output")

    result = np.concatenate((encoded, encoded[:_REPETITION_BITS])).astype(np.uint8, copy=False)
    result.setflags(write=False)
    return result


def nr_pbch_bch_decode(
    rate_matched_llrs: NDArray[np.generic],
    *,
    device: str = "cpu",
    list_size: int = 8,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Decode PBCH BCH LLRs after physical-channel descrambling.

    The final dimension must contain the 864 TS 38.212 rate-matched LLRs,
    with positive values favoring bit one (Sionna's logit convention). Any
    leading dimensions are treated as batch dimensions. This combines PBCH's
    352 repeated-bit LLRs, then applies Sionna's downlink Polar rate recovery,
    CRC-aided SCL decoding, and CRC24C check. The returned pair contains the
    recovered 32-bit transport block and a per-block CRC-validity flag.

    Physical-channel descrambling, demodulation, channel estimation, and the
    PBCH payload generation/interleaving steps are outside this function.
    This requires the optional pinned ``sionna==2.2.0`` dependency.
    """
    llrs = _rate_matched_llrs(rate_matched_llrs)
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    if not device:
        raise ValueError("device must not be empty")
    if isinstance(list_size, bool) or not isinstance(list_size, int) or list_size <= 0:
        raise ValueError("list_size must be a positive integer")

    decoder = _polar_decoder(device, list_size)
    import torch

    mother_llrs = np.array(llrs[..., :_POLAR_MOTHER_BITS], copy=True)
    mother_llrs[..., :_REPETITION_BITS] += llrs[..., _POLAR_MOTHER_BITS:]
    np.clip(mother_llrs, -100.0, 100.0, out=mother_llrs)
    llr_tensor = torch.as_tensor(mother_llrs, dtype=torch.float32, device=device)
    decoded, crc_valid = decoder(llr_tensor)
    payload = decoded.to(torch.uint8).detach().cpu().numpy().copy()
    crc_status = crc_valid.detach().cpu().numpy().astype(np.bool_, copy=True)
    expected_shape = (*llrs.shape[:-1], _BCH_INPUT_BITS)
    if payload.shape != expected_shape or crc_status.shape != llrs.shape[:-1]:
        raise RuntimeError("pinned Sionna Polar decoder returned an unexpected output shape")
    payload.setflags(write=False)
    crc_status.setflags(write=False)
    return payload, crc_status


def _binary_payload(bits: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(bits)
    if array.shape != (_BCH_INPUT_BITS,):
        raise ValueError("scrambled_transport_block must have shape (32,)")
    if array.dtype.kind not in "biu" or np.any(array < 0) or np.any(array > 1):
        raise ValueError("scrambled_transport_block must contain only binary values")
    return np.asarray(array, dtype=np.uint8)


def _rate_matched_llrs(llrs: NDArray[np.generic]) -> NDArray[np.float64]:
    array = np.asarray(llrs)
    if array.ndim < 1 or array.shape[-1] != _RATE_MATCHED_BITS:
        raise ValueError("rate_matched_llrs must have final dimension 864")
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError("rate_matched_llrs must contain finite real values")
    return np.asarray(array, dtype=np.float64)


@lru_cache(maxsize=4)
def _polar_encoder(device: str) -> Any:
    try:
        installed_version = version("sionna")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "PBCH BCH encoding requires the optional Sionna 2.2.0 dependency; "
            "install the project 'sionna' extra"
        ) from error
    if installed_version != "2.2.0":
        raise RuntimeError(f"PBCH BCH encoding requires Sionna 2.2.0, found {installed_version}")

    try:
        from sionna.phy.fec.polar import Polar5GEncoder
    except ImportError as error:
        raise RuntimeError(
            "PBCH BCH encoding requires the optional Sionna 2.2.0 dependency; "
            "install the project 'sionna' extra"
        ) from error

    # With K=32+CRC24C=56 and E=864, TS 38.212 selects n=9, N=512.
    # Asking Sionna for E=N=512 yields the standard sub-block-interleaved
    # mother codeword. PBCH's E=864 rate matcher then repeats its prefix.
    return Polar5GEncoder(
        _BCH_INPUT_BITS,
        _POLAR_MOTHER_BITS,
        channel_type="downlink",
        device=device,
    )


@lru_cache(maxsize=8)
def _polar_decoder(device: str, list_size: int) -> Any:
    encoder = _polar_encoder(device)
    try:
        from sionna.phy.fec.polar import Polar5GDecoder
    except ImportError as error:
        raise RuntimeError(
            "PBCH BCH decoding requires the optional Sionna 2.2.0 dependency; "
            "install the project 'sionna' extra"
        ) from error
    return Polar5GDecoder(
        encoder,
        dec_type="SCL",
        list_size=list_size,
        return_crc_status=True,
        device=device,
    )

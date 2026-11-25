from __future__ import annotations

import builtins
from importlib.metadata import PackageNotFoundError

import integrations.sionna.pbch_bch as pbch
import numpy as np
import pytest


def test_scl_list_size_requires_bounded_power_of_two() -> None:
    assert pbch.validate_pbch_scl_list_size(np.int64(16)) == 16
    for value, error, message in [
        (True, TypeError, "must be an integer"),
        (np.bool_(False), TypeError, "must be an integer"),
        (0, ValueError, "must be positive"),
        (3, ValueError, "power of two"),
        (128, ValueError, "at most 64"),
    ]:
        with pytest.raises(error, match=message):
            pbch.validate_pbch_scl_list_size(value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (np.zeros(31, dtype=np.uint8), r"shape \(32,\)"),
        (np.full(32, 2, dtype=np.uint8), "binary values"),
        (np.full(32, -1, dtype=np.int8), "binary values"),
        (np.zeros(32, dtype=np.float32), "binary values"),
    ],
)
def test_payload_validator_rejects_wrong_shape_and_nonbinary_values(
    payload: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pbch._binary_payload(payload)


@pytest.mark.parametrize(
    ("llrs", "message"),
    [
        (np.zeros(863), "final dimension 864"),
        (np.full(864, np.inf), "finite real values"),
        (np.zeros(864, dtype=np.complex64), "finite real values"),
    ],
)
def test_llr_validator_rejects_wrong_shape_and_nonreal_values(
    llrs: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pbch._rate_matched_llrs(llrs)


@pytest.mark.parametrize("device", [None, 3])
def test_encoder_and_decoder_require_string_device(device: object) -> None:
    with pytest.raises(TypeError, match="device must be a string"):
        pbch.nr_pbch_bch_encode(np.zeros(32, dtype=np.uint8), device=device)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="device must be a string"):
        pbch.nr_pbch_bch_decode(np.zeros(864), device=device)  # type: ignore[arg-type]


def test_encoder_and_decoder_reject_empty_device() -> None:
    with pytest.raises(ValueError, match="device must not be empty"):
        pbch.nr_pbch_bch_encode(np.zeros(32, dtype=np.uint8), device="")
    with pytest.raises(ValueError, match="device must not be empty"):
        pbch.nr_pbch_bch_decode(np.zeros(864), device="")


@pytest.mark.parametrize(
    ("encoder_output", "message"),
    [
        (np.zeros(511, dtype=np.uint8), "unexpected mother-code length"),
        (np.full(512, 2, dtype=np.uint8), "non-binary output"),
    ],
)
def test_encoder_checks_pinned_polar_output(
    monkeypatch: pytest.MonkeyPatch, encoder_output: np.ndarray, message: str
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        pbch, "_polar_encoder", lambda _device: lambda _payload: torch.as_tensor(encoder_output)
    )

    with pytest.raises(RuntimeError, match=message):
        pbch.nr_pbch_bch_encode(np.zeros(32, dtype=np.uint8))


def test_encoder_builds_immutable_rate_matched_output(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mother = np.arange(512, dtype=np.uint8) % 2
    monkeypatch.setattr(
        pbch, "_polar_encoder", lambda _device: lambda _payload: torch.as_tensor(mother)
    )

    encoded = pbch.nr_pbch_bch_encode(np.zeros(32, dtype=np.uint8))

    assert encoded.shape == (864,)
    np.testing.assert_array_equal(encoded[512:], mother[:352])
    assert not encoded.flags.writeable


def test_decoder_checks_pinned_polar_output_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    def wrong_shape(_llrs: object) -> tuple[object, object]:
        return torch.zeros((31,), dtype=torch.uint8), torch.tensor(True)

    monkeypatch.setattr(pbch, "_polar_decoder", lambda *_args: wrong_shape)
    with pytest.raises(RuntimeError, match="unexpected output shape"):
        pbch.nr_pbch_bch_decode(np.zeros(864))


def test_decoder_returns_immutable_payload_and_crc(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    payload = np.arange(64, dtype=np.uint8).reshape(2, 32) % 2

    def decode(llrs: object) -> tuple[object, object]:
        assert tuple(llrs.shape) == (2, 512)  # type: ignore[attr-defined]
        return torch.as_tensor(payload), torch.tensor([True, False])

    monkeypatch.setattr(pbch, "_polar_decoder", lambda *_args: decode)
    decoded, crc_valid = pbch.nr_pbch_bch_decode(np.zeros((2, 864)), list_size=2)
    np.testing.assert_array_equal(decoded, payload)
    np.testing.assert_array_equal(crc_valid, [True, False])
    assert not decoded.flags.writeable
    assert not crc_valid.flags.writeable


@pytest.mark.parametrize(
    ("installed_version", "message"),
    [
        (None, "optional Sionna 2.2.0 dependency"),
        ("2.1.0", "requires Sionna 2.2.0, found 2.1.0"),
    ],
)
def test_encoder_requires_the_pinned_optional_sionna_version(
    monkeypatch: pytest.MonkeyPatch, installed_version: str | None, message: str
) -> None:
    pbch._polar_encoder.cache_clear()
    if installed_version is None:

        def missing_version() -> str:
            raise PackageNotFoundError("sionna")

        monkeypatch.setattr(pbch, "sionna_phy_version", missing_version)
    else:
        monkeypatch.setattr(pbch, "sionna_phy_version", lambda: installed_version)

    with pytest.raises(RuntimeError, match=message):
        pbch._polar_encoder("cpu")
    pbch._polar_encoder.cache_clear()


def test_encoder_converts_missing_polar_module_to_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pbch._polar_encoder.cache_clear()
    monkeypatch.setattr(pbch, "sionna_phy_version", lambda: "2.2.0")
    original_import = builtins.__import__

    def without_polar(name: str, *args: object, **kwargs: object) -> object:
        if name == "sionna.phy.fec.polar":
            raise ImportError("deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_polar)
    with pytest.raises(RuntimeError, match="optional Sionna 2.2.0 dependency"):
        pbch._polar_encoder("cpu")
    pbch._polar_encoder.cache_clear()


def test_decoder_converts_missing_polar_decoder_to_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pbch._polar_decoder.cache_clear()
    monkeypatch.setattr(pbch, "_polar_encoder", lambda _device: object())
    original_import = builtins.__import__

    def without_polar(name: str, *args: object, **kwargs: object) -> object:
        if name == "sionna.phy.fec.polar":
            raise ImportError("deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_polar)
    with pytest.raises(RuntimeError, match="decoding requires the optional Sionna"):
        pbch._polar_decoder("cpu", 8)
    pbch._polar_decoder.cache_clear()


def test_optional_encoder_and_decoder_constructors_use_pinned_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pbch._polar_encoder.cache_clear()
    pbch._polar_decoder.cache_clear()
    monkeypatch.setattr(pbch, "sionna_phy_version", lambda: "2.2.0")
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class PolarEncoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(("encoder", args, kwargs))

    class PolarDecoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(("decoder", args, kwargs))

    polar_module = type(
        "PolarModule",
        (),
        {"Polar5GEncoder": PolarEncoder, "Polar5GDecoder": PolarDecoder},
    )()
    original_import = builtins.__import__

    def fake_polar_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sionna.phy.fec.polar":
            return polar_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_polar_import)
    encoder = pbch._polar_encoder("cpu")
    decoder = pbch._polar_decoder("cpu", 4)

    assert isinstance(encoder, PolarEncoder)
    assert isinstance(decoder, PolarDecoder)
    assert calls[0] == (
        "encoder",
        (32, 512),
        {"channel_type": "downlink", "device": "cpu"},
    )
    assert calls[1] == (
        "decoder",
        (encoder,),
        {"dec_type": "SCL", "list_size": 4, "return_crc_status": True, "device": "cpu"},
    )
    pbch._polar_encoder.cache_clear()
    pbch._polar_decoder.cache_clear()

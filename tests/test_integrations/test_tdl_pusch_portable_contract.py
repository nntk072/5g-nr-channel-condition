import builtins
import sys
from types import ModuleType, SimpleNamespace

import integrations.sionna.tdl_pusch as tdl_pusch
import pytest
import torch


def _install_sionna_modules(monkeypatch, *, time_domain=False):
    sionna = ModuleType("sionna")
    phy = ModuleType("sionna.phy")
    channel = ModuleType("sionna.phy.channel")
    tr38901 = ModuleType("sionna.phy.channel.tr38901")
    nr = ModuleType("sionna.phy.nr")
    if time_domain:

        class TimeChannel:
            def __init__(self, model, **kwargs):
                self.model = model
                self.options = kwargs

            def __call__(self, waveform):
                return waveform + 1, torch.ones_like(waveform)

        channel.TimeChannel = TimeChannel
        channel.time_lag_discrete_time_channel = lambda bandwidth, delay: (-1, 4)
    else:

        class OFDMChannel:
            def __init__(self, model, grid, **kwargs):
                self.grid = grid

            def __call__(self, grid):
                return grid + 1, torch.ones_like(grid)

        channel.OFDMChannel = OFDMChannel

    class TDL:
        def __init__(self, **kwargs):
            self.options = kwargs

    class Transmitter:
        def __init__(self, config, **kwargs):
            self.resource_grid = SimpleNamespace(
                bandwidth=30.72e6,
                num_time_samples=1024,
                cyclic_prefix_length=torch.tensor([4, 4]),
                fft_size=64,
                ofdm_symbol_duration=1 / 30e3,
            )

        def __call__(self, batch_size):
            return torch.ones((batch_size, 2), dtype=torch.complex64), torch.ones(
                (batch_size, 4), dtype=torch.int32
            )

    class Receiver:
        def __init__(self, transmitter, **kwargs):
            self._transmitter = transmitter
            self._estimator = kwargs.get("channel_estimator")
            self.options = kwargs

        def __call__(self, received, variance):
            return torch.ones((received.shape[0], 4)), torch.ones(
                received.shape[0], dtype=torch.bool
            )

    tr38901.TDL = TDL
    nr.PUSCHTransmitter = Transmitter
    nr.PUSCHReceiver = Receiver
    for name, module in (
        ("sionna", sionna),
        ("sionna.phy", phy),
        ("sionna.phy.channel", channel),
        ("sionna.phy.channel.tr38901", tr38901),
        ("sionna.phy.nr", nr),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return channel, tr38901, nr


@pytest.mark.parametrize(
    ("shape", "variance", "real_seed", "imag_seed", "message"),
    [
        ((), 1.0, 1, 2, "shape"),
        ((2, 0), 1.0, 1, 2, "shape"),
        ((True,), 1.0, 1, 2, "shape"),
        ((2,), True, 1, 2, "real scalar"),
        ((2,), float("nan"), 1, 2, "finite and nonnegative"),
        ((2,), -1.0, 1, 2, "finite and nonnegative"),
        ((2,), 1e100, 1, 2, "representable as finite"),
        ((2,), 1e-100, 1, 2, "representable as nonzero"),
        ((2,), 1.0, True, 2, "real_seed"),
        ((2,), 1.0, 1, -1, "imag_seed"),
    ],
)
def test_seeded_awgn_rejects_invalid_contract(shape, variance, real_seed, imag_seed, message):
    with pytest.raises(ValueError, match=message):
        tdl_pusch.seeded_complex_awgn(
            shape,
            device="cpu",
            noise_variance=variance,
            real_seed=real_seed,
            imag_seed=imag_seed,
        )


def test_seeded_awgn_has_reproducible_component_streams_and_prefixes():
    short = tdl_pusch.seeded_complex_awgn(
        (2, 3), device="cpu", noise_variance=0.4, real_seed=10, imag_seed=20
    )
    repeated = tdl_pusch.seeded_complex_awgn(
        (2, 3), device="cpu", noise_variance=0.4, real_seed=10, imag_seed=20
    )
    long = tdl_pusch.seeded_complex_awgn(
        (2, 7), device="cpu", noise_variance=0.4, real_seed=10, imag_seed=20
    )
    torch.testing.assert_close(short, repeated)
    torch.testing.assert_close(short, long[:, :3])
    assert not torch.equal(short.real, short.imag)


def test_optional_decoder_and_time_link_report_missing_sionna(monkeypatch):
    original_import = builtins.__import__

    def import_without_sionna(name, *args, **kwargs):
        if name.startswith("sionna"):
            raise ImportError("hidden optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sionna)
    with pytest.raises(RuntimeError, match="optional Sionna 2.2 PHY"):
        tdl_pusch.SionnaPUSCHLinkDecoder(SimpleNamespace())
    with pytest.raises(RuntimeError, match="optional Sionna 2.2 PHY"):
        tdl_pusch.SionnaTDLTimePUSCHLink(
            SimpleNamespace(),
            model="A",
            delay_spread_s=1e-7,
            carrier_frequency_hz=3.5e9,
            speed_mps=1.0,
        )


def test_decoder_builds_crc_receiver_and_forwards_decode_inputs(monkeypatch):
    _, _, nr = _install_sionna_modules(monkeypatch)
    receiver = SimpleNamespace(_transmitter=object(), _estimator=object(), device="cpu")
    decoder = tdl_pusch.SionnaPUSCHLinkDecoder(receiver)
    assert decoder._decoder.options["input_domain"] == "freq"
    grid = torch.ones((2, 3), dtype=torch.complex64)
    variance = torch.tensor(0.1)
    bits, crc = decoder(grid, variance)
    assert bits.shape == (2, 4)
    assert crc.shape == (2,)


def test_time_link_checks_receiver_contract_cp_and_transmit_seeds(monkeypatch):
    _, _, nr = _install_sionna_modules(monkeypatch, time_domain=True)
    receiver = SimpleNamespace(
        config=SimpleNamespace(num_layers=1, num_antenna_ports=1),
        device="cpu",
        _transmitter=SimpleNamespace(_pusch_configs=object()),
        _estimator=object(),
    )
    with pytest.raises(ValueError, match="one layer"):
        tdl_pusch.SionnaTDLTimePUSCHLink(
            SimpleNamespace(
                **{
                    **receiver.__dict__,
                    "config": SimpleNamespace(num_layers=2, num_antenna_ports=1),
                }
            ),
            model="A",
            delay_spread_s=1e-7,
            carrier_frequency_hz=3.5e9,
            speed_mps=1.0,
        )
    with pytest.raises(ValueError, match="num_rx_ant"):
        tdl_pusch.SionnaTDLTimePUSCHLink(
            receiver,
            model="A",
            delay_spread_s=1e-7,
            carrier_frequency_hz=3.5e9,
            speed_mps=1.0,
            num_rx_ant=0,
        )
    with pytest.raises(TypeError, match="allow_cp_overrun"):
        tdl_pusch.SionnaTDLTimePUSCHLink(
            receiver,
            model="A",
            delay_spread_s=1e-7,
            carrier_frequency_hz=3.5e9,
            speed_mps=1.0,
            allow_cp_overrun=1,
        )
    with pytest.raises(ValueError, match="channel memory exceeds"):
        tdl_pusch.SionnaTDLTimePUSCHLink(
            receiver,
            model="A",
            delay_spread_s=1e-7,
            carrier_frequency_hz=3.5e9,
            speed_mps=1.0,
        )
    link = tdl_pusch.SionnaTDLTimePUSCHLink(
        receiver,
        model="A",
        delay_spread_s=1e-7,
        carrier_frequency_hz=3.5e9,
        speed_mps=1.0,
        allow_cp_overrun=True,
    )
    with pytest.raises(ValueError, match="batch_size"):
        link.transmit_channel(0, seed=1)
    with pytest.raises(ValueError, match="seed must be a nonnegative"):
        link.transmit_channel(1, seed=-1)
    with pytest.raises(ValueError, match="leave room"):
        link.transmit_channel(1, seed=2**63 - 1)
    monkeypatch.setattr(tdl_pusch, "call_with_sionna_seed", lambda device, seed, call: call())
    waveform, bits, received, truth = link.transmit_channel(2, seed=3)
    assert waveform.shape == (2, 2)
    assert bits.shape == (2, 4)
    assert received is not None and truth is not None
    bits, crc = link.decode(received, torch.tensor(0.1))
    assert bits.shape == (2, 4)
    assert crc.shape == (2,)


def test_generator_rejects_noise_extremes_and_missing_optional_dependencies(monkeypatch):
    receiver_contract = SimpleNamespace(config=SimpleNamespace(num_layers=1, num_antenna_ports=1))
    with pytest.raises(ValueError, match="noise_variance must be a real scalar"):
        tdl_pusch.generate_tdl_pusch_batch(
            receiver_contract,
            model="A",
            batch_size=1,
            seed=0,
            carrier_frequency_hz=3.5e9,
            delay_spread_s=1e-7,
            noise_variance="not numeric",
        )
    with pytest.raises(ValueError, match="finite float32"):
        tdl_pusch.generate_tdl_pusch_batch(
            SimpleNamespace(config=SimpleNamespace(num_layers=1, num_antenna_ports=1)),
            model="A",
            batch_size=1,
            seed=0,
            carrier_frequency_hz=3.5e9,
            delay_spread_s=1e-7,
            noise_variance=1e100,
        )
    with pytest.raises(ValueError, match="nonzero float32"):
        tdl_pusch.generate_tdl_pusch_batch(
            SimpleNamespace(config=SimpleNamespace(num_layers=1, num_antenna_ports=1)),
            model="A",
            batch_size=1,
            seed=0,
            carrier_frequency_hz=3.5e9,
            delay_spread_s=1e-7,
            noise_variance=1e-100,
        )

    original_import = builtins.__import__

    def import_without_sionna(name, *args, **kwargs):
        if name.startswith("sionna"):
            raise ImportError("hidden optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sionna)
    with pytest.raises(RuntimeError, match="optional Sionna 2.2 PHY dependencies"):
        call = SimpleNamespace(
            config=SimpleNamespace(num_layers=1, num_antenna_ports=1),
            resource_grid=SimpleNamespace(),
            device="cpu",
        )
        tdl_pusch.generate_tdl_pusch_batch(
            call,
            model="A",
            batch_size=1,
            seed=0,
            carrier_frequency_hz=3.5e9,
            delay_spread_s=1e-7,
            noise_variance=0.0,
        )


def test_generator_returns_seeded_observation_and_private_channel_truth(monkeypatch):
    channel, tr38901, _ = _install_sionna_modules(monkeypatch)
    receiver = SimpleNamespace(
        config=SimpleNamespace(num_layers=1, num_antenna_ports=1),
        resource_grid=SimpleNamespace(
            effective_subcarrier_ind=torch.tensor([1, 2]),
            ofdm_symbol_duration=1 / 30e3,
        ),
        device="cpu",
    )
    receiver.transmit = lambda batch_size, seed: (
        torch.ones((batch_size, 2), dtype=torch.complex64),
        torch.ones((batch_size, 4), dtype=torch.int32),
    )
    monkeypatch.setattr(tdl_pusch, "call_with_sionna_seed", lambda device, seed, call: call())
    monkeypatch.setattr(tdl_pusch, "call_with_sionna_seed", lambda device, seed, call: call())
    batch = tdl_pusch.generate_tdl_pusch_batch(
        receiver,
        model="B",
        batch_size=2,
        seed=5,
        carrier_frequency_hz=3.5e9,
        delay_spread_s=100e-9,
        noise_variance=0.1,
        num_rx_ant=2,
    )
    assert batch.transmit_grid.shape == (2, 2)
    assert batch.received_grid.shape == (2, 2)
    assert batch.channel_truth.shape == (2, 2)
    assert batch.noise_variance.item() == pytest.approx(0.1)
    assert batch.metadata["geometric_los_label"] is None
    assert batch.metadata["sampling_frequency_hz"] == pytest.approx(30e3)
    assert tr38901.TDL is not None and channel.OFDMChannel is not None


def test_generator_rejects_unrepresentable_seed_and_unavailable_cuda(monkeypatch):
    channel, tr38901, _ = _install_sionna_modules(monkeypatch)
    receiver = SimpleNamespace(
        config=SimpleNamespace(num_layers=1, num_antenna_ports=1),
        resource_grid=SimpleNamespace(effective_subcarrier_ind=[0], ofdm_symbol_duration=1.0),
        device="cpu",
        transmit=lambda batch_size, seed: (
            torch.ones((batch_size, 1)),
            torch.ones((batch_size, 1)),
        ),
    )
    common = {
        "model": "A",
        "batch_size": 1,
        "carrier_frequency_hz": 3.5e9,
        "delay_spread_s": 0.0,
        "noise_variance": 0.0,
    }
    with pytest.raises(ValueError, match="leave room"):
        tdl_pusch.generate_tdl_pusch_batch(receiver, seed=2**63 - 2, **common)
    receiver.device = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device requested but unavailable"):
        tdl_pusch.generate_tdl_pusch_batch(receiver, seed=0, **common)


def test_batch_to_records_preserves_effective_cfr_and_batch_provenance(monkeypatch):
    class Target:
        @staticmethod
        def versioned(**kwargs):
            return kwargs

    class Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Observation:
        @staticmethod
        def from_receiver_output(**kwargs):
            return kwargs

    monkeypatch.setattr(tdl_pusch, "NRChannelEstimate", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(tdl_pusch, "NRObservation", Observation)
    monkeypatch.setattr(tdl_pusch, "NRChannelRecord", Record)
    monkeypatch.setattr(tdl_pusch, "TargetDefinition", Target)
    receiver = SimpleNamespace(
        _torch=torch,
        resource_grid=SimpleNamespace(effective_subcarrier_ind=torch.tensor([1, 3])),
        config=object(),
        estimate=lambda grid, variance: SimpleNamespace(
            h_hat=torch.ones((2, 1, 1, 1, 1, 4), dtype=torch.complex64),
            error_variance=torch.tensor([0.2, 0.3]),
            pilot_mask=torch.ones((1, 4), dtype=torch.bool),
            metadata={"receiver": "ideal"},
        ),
    )
    truth = torch.arange(16, dtype=torch.float32).to(torch.complex64).reshape(2, 1, 1, 1, 1, 1, 8)
    batch = tdl_pusch.SionnaTDLPUSCHBatch(
        transmit_grid=torch.ones((2, 1)),
        transmit_bits=torch.ones((2, 2)),
        received_grid=torch.ones((2, 1, 1, 1, 8), dtype=torch.complex64),
        channel_truth=truth,
        noise_variance=torch.tensor(0.1),
        metadata={"profile_proxy": "A", "geometric_los_label": None},
    )
    records = tdl_pusch.tdl_pusch_records(receiver, batch, sample_ids=("s0", "s1"))
    assert len(records) == 2
    assert [record.provenance["batch_index"] for record in records] == [0, 1]
    for index, record in enumerate(records):
        assert record.targets["channel_cfr"].shape[-1] == 2
        torch.testing.assert_close(
            record.targets["channel_cfr"].flatten(),
            truth[index : index + 1]
            .squeeze(4)
            .unsqueeze(4)
            .index_select(-1, torch.tensor([1, 3]))
            .flatten(),
        )
        assert record.target_masks["channel_cfr"].all()

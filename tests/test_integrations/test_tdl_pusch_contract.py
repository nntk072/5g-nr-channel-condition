from types import SimpleNamespace

import pytest
from integrations.sionna.tdl_pusch import generate_tdl_pusch_batch, tdl_pusch_records


def receiver(num_layers=1, num_antenna_ports=1):
    return SimpleNamespace(
        config=SimpleNamespace(
            num_layers=num_layers,
            num_antenna_ports=num_antenna_ports,
        )
    )


def call(**overrides):
    values = {
        "receiver": receiver(),
        "model": "A",
        "batch_size": 2,
        "seed": 1,
        "carrier_frequency_hz": 3.5e9,
        "delay_spread_s": 100e-9,
        "noise_variance": 0.01,
    }
    values.update(overrides)
    return generate_tdl_pusch_batch(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"num_rx_ant": 0}, "num_rx_ant"),
        ({"seed": -1}, "seed"),
        ({"noise_variance": -1.0}, "noise_variance"),
        ({"carrier_frequency_hz": 0.0}, "carrier_frequency_hz"),
        ({"delay_spread_s": -1.0}, "delay_spread_s"),
        ({"receiver": receiver(num_antenna_ports=2)}, "one transmit antenna port"),
    ],
)
def test_tdl_pusch_generator_rejects_invalid_inputs_before_optional_import(overrides, message):
    with pytest.raises(ValueError, match=message):
        call(**overrides)


@pytest.mark.parametrize("sample_ids", [(), ("same", "same")])
def test_tdl_pusch_record_conversion_requires_unique_ids_for_every_example(sample_ids):
    batch = SimpleNamespace(received_grid=SimpleNamespace(shape=(2, 1, 1, 14, 24)))
    with pytest.raises(ValueError, match="uniquely identify every batch example"):
        tdl_pusch_records(receiver(), batch, sample_ids=sample_ids)

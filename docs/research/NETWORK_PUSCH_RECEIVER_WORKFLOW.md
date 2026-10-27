# Sample-domain network waveform to PUSCH receiver

`integrations.sionna.network_receiver.decode_noisy_network_pusch` connects a
bounded aligned network waveform and its receiver-specific AWGN result to the
existing Sionna PUSCH demodulator/decoder. It checks that per-link contributions
sum to the clean capture, that the noisy capture equals clean capture plus the
retained noise samples, and that the sample rate matches the configured FFT
grid. The exact input/output tensor digests, receiver metadata digest, device,
dimensions, and CRC totals are retained in a schema-validated v1 report.

The caller must provide receiver-domain noise variance in watts per resource
element. The adapter retains sample-domain variance separately and does not
convert between them. Until the OFDM normalization and variance conversion are
verified for each supported receiver convention, callers must supply the
receiver-side value from a declared calibration or independently validated
calculation. Sample alignment, synchronization, and acquisition are also
caller-controlled.

```python
import torch

from integrations.sionna.network_phy import (
    add_independent_receiver_waveform_awgn,
    compose_received_waveforms,
)
from integrations.sionna.network_receiver import decode_noisy_network_pusch
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from receivers.nr import NRPUSCHConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"
receiver = SionnaPUSCHLSReceiver(
    NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30), device=device
)
tx_grid, _tx_bits = receiver.transmit(batch_size=1, seed=12345)
tx_waveform = receiver.modulate_slot_grid(tx_grid)

# Axes: [batch, receiver, transmitter, receive_port, sample].
links = tx_waveform[:, None, :, :1, :]
composition = compose_received_waveforms(
    links,
    link_start_samples=((0,),),
    capture_num_samples=tx_waveform.shape[-1],
    sample_rate_hz=receiver.resource_grid.fft_size * 30_000,
)
noisy = add_independent_receiver_waveform_awgn(
    composition, (1e-7,), seed=54321
)
result = decode_noisy_network_pusch(
    composition,
    noisy,
    receiver,
    noise_variance_w_per_re=(1e-6,),  # caller-declared W/RE convention
)
print(result.report["status"], result.crc_ok)
```

The receiver API and native V100 smoke prove software-path integration for one
small synthetic PUSCH configuration only. Reports remain
`OBSERVED_NOT_QUALIFIED`; no field or deployment qualification follows. The
strict report contract is
`datasets/schema/network-pusch-receiver-decode.v1.schema.json`.

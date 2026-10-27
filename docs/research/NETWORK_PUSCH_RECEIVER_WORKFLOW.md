# Sample-domain network waveform to PUSCH receiver

`integrations.sionna.network_receiver.decode_noisy_network_pusch` connects a
bounded aligned network waveform and its receiver-specific AWGN result to the
existing Sionna PUSCH demodulator/decoder. It checks that per-link contributions
sum to the clean capture, that the noisy capture equals clean capture plus the
retained noise samples, and that the sample rate matches the configured FFT
grid. The exact input/output tensor digests, receiver metadata digest, device,
dimensions, and CRC totals are retained in a schema-validated v3 report. The
report also records the nonzero support and memory of each declared causal
per-link FIR relative to the receiver's configured cyclic prefix, bound to the
exact tap tensor digest. This is explicitly a filter-support diagnostic only:
it excludes link start offsets, fractional-delay interpolation support, and
receiver acquisition, so it does not establish total channel memory, ISI, or
receiver qualification. The previously published v1 and v2 schemas remain
unchanged.

For the pinned Sionna PHY 2.2.0 receiver, the adapter converts sample-domain
variance to per-RE variance using the unitary FFT identity. Sionna applies
`1/sqrt(N)` in both FFT and IFFT; CP removal selects iid samples, phase
compensation has unit magnitude, and bin reordering is a permutation. Thus
proper-complex iid sample noise with variance `sigma²` remains at variance
`sigma²` per resource element. A native 512-slot noise-only check measures the
demodulated variance against that value. Other receiver versions still require
an explicit caller-supplied W/RE profile. Sample alignment, synchronization,
and acquisition remain caller-controlled.

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
    noise_variance_w_per_re=None,  # uses the pinned Sionna 2.2.0 unitary-FFT identity
)
print(result.report["status"], result.crc_ok)
```

The receiver API and native V100 smoke prove software-path integration for one
small synthetic PUSCH configuration only. Reports remain
`OBSERVED_NOT_QUALIFIED`; no field or deployment qualification follows. The
strict report contract is
`datasets/schema/network-pusch-receiver-decode.v3.schema.json`.

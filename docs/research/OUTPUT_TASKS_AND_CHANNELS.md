# Model outputs and channel families

This project contains several related experiments. Their labels and outputs are
different, so results must not be combined under one generic “channel
condition” metric.

| Task | Model output | Channel/data source | Current evidence |
|---|---|---|---|
| Main channel-condition classifier | Binary probability/class for geometric direct-path LOS vs. NLOS | Six Sionna RT scene families, converted to receiver-side 5G NR PUSCH/DMRS observations | Six-scene leave-one-scene-out pilot. Large CNN macro AUROC 0.814; raw-IQ MLP 0.806. Only 40 test links and 4–20 spatial groups per held-out scene; exploratory, synthetic, and not independently geometry/material-qualified. |
| Standardized profile proxy | Binary A/B/C vs. D/E profile grouping | Sionna 3GPP TDL-A–E and CDL-A–E channel-profile data | Leave-one-profile-out MLP profile-macro recall 0.932 over five profile groups. The proxy label identifies a synthetic profile grouping; it is not LOS/NLOS or a measured condition. Each held-out profile has only one proxy class, so pooled AUROC is undefined. |
| Multipath model-order estimation | Estimated path count (0–3) and a fitted CFR from receiver observations | Seeded analytical multipath channels | Exact order selected in 791/960 trials (82.4%); accuracy was 94.6%, 91.3%, 81.3%, and 62.5% for true orders 0–3. Synthetic estimator stress only. |
| Measured CSI-magnitude forecasting | Baseline forecast of future magnitude tensor [O-RU, Rx antenna, subcarrier]; Ridge regression of per-O-RU/antenna subcarrier-mean magnitude in dB | ETH Zurich CAEZ-5G indoor, outdoor, and hallway PUSCH CSI; official preprocessed magnitude features | Full-tensor indoor persistence NRMSE 0.708 vs. trend 1.478. Compact-target zero-shot indoor-trained Ridge NRMSE: indoor 0.807 vs. persistence 1.047; outdoor 0.800 vs. 0.915; hallway 0.945 vs. 0.275. Ridge transfers better than persistence indoors/outdoors, but fails in hallway. One chronological tail per campaign; site/UE/protocol confounded; no phase or LOS/NLOS truth. Exploratory only. See [cross-environment report](results/caez-cross-environment-summary-forecast.json). |

## How TDL/CDL fit

TDL-A–E and CDL-A–E are included in standardized channel-profile validation
and the separate profile-proxy experiment. TDL profiles also drive receiver
and PUSCH link checks, including TDL-A BLER/mobility sweeps. The headline
LOS/NLOS classifier instead uses geometric labels from Sionna RT scenes. These
families do not share one trained classifier or one interchangeable label.

The current paper-ready story should therefore keep the RT LOS/NLOS result as
the main classification task, use the profile-proxy and analytical
multipath-order results as explicitly separate simulation diagnostics, and
describe CAEZ as an exploratory measured magnitude-forecast pilot: the indoor-trained Ridge model beats persistence on the compact mean-magnitude dB summary indoors and outdoors, loses in the hallway, and the full-tensor indoor baseline favors persistence over linear trend. None of these pilots alone establishes physical LOS/NLOS performance or a general
model advantage.

Detailed results: [six-scene LOSO](results/mvp-six-scene-dmrs-losso-seed20261022.json),
[capacity LOSO](results/mvp-capacity-losso-20261022.json),
[profile holdout](results/profile-holdout-baselines-seed20261021.json),
[multipath order](results/multipath-order-stress-seed20261022.json), and
[CAEZ forecast](results/caez-indoor-magnitude-forecast.json).

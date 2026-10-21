# Model outputs and channel families

This project contains several related experiments. Their labels and outputs are
different, so results must not be combined under one generic “channel
condition” metric.

| Task | Model output | Channel/data source | Current evidence |
|---|---|---|---|
| Main channel-condition classifier | Binary probability/class for geometric direct-path LOS vs. NLOS | Six Sionna RT scene families, converted to receiver-side 5G NR PUSCH/DMRS observations | Six-scene leave-one-scene-out pilot. Large CNN macro AUROC 0.814; raw-IQ MLP 0.806. Only 40 test links and 4–20 spatial groups per held-out scene; exploratory, synthetic, and not independently geometry/material-qualified. |
| Standardized profile proxy | Binary A/B/C vs. D/E profile grouping | Sionna 3GPP TDL-A–E and CDL-A–E channel-profile data | Leave-one-profile-out MLP profile-macro recall 0.932 over five profile groups. The proxy label identifies a synthetic profile grouping; it is not LOS/NLOS or a measured condition. Each held-out profile has only one proxy class, so pooled AUROC is undefined. |
| Multipath model-order estimation | Estimated path count (0–3) and a fitted CFR from receiver observations | Seeded analytical multipath channels | Exact order selected in 791/960 trials (82.4%); accuracy was 94.6%, 91.3%, 81.3%, and 62.5% for true orders 0–3. Synthetic estimator stress only. |
| Measured CSI-magnitude forecasting | Full future magnitude tensor [O-RU, Rx antenna, subcarrier] with baseline predictors; learned regression of each O-RU/antenna's subcarrier-mean magnitude in dB | ETH Zurich CAEZ-5G indoor PUSCH CSI, preprocessed magnitude-only features | On 67,717 one-second test pairs, full-tensor persistence NRMSE is 0.708 vs. trend 1.478. For the compact summary target, Ridge RMSE normalized by held-out target standard deviation is 0.807 vs. persistence 1.047 and trend 2.577; trained on 237,235 earlier pairs with a 10% temporal gap. Single indoor campaign, no phase or LOS/NLOS truth; exploratory only. |

## How TDL/CDL fit

TDL-A–E and CDL-A–E are included in standardized channel-profile validation
and the separate profile-proxy experiment. TDL profiles also drive receiver
and PUSCH link checks, including TDL-A BLER/mobility sweeps. The headline
LOS/NLOS classifier instead uses geometric labels from Sionna RT scenes. These
families do not share one trained classifier or one interchangeable label.

The current paper-ready story should therefore keep the RT LOS/NLOS result as
the main classification task, use the profile-proxy and analytical
multipath-order results as explicitly separate simulation diagnostics, and
describe CAEZ as an initial measured magnitude-forecast pilot: the small Ridge model beats persistence only on the compact mean-magnitude dB summary, while the full-tensor baseline still favors persistence. None of these pilots alone establishes physical LOS/NLOS performance or a general
model advantage.

Detailed results: [six-scene LOSO](results/mvp-six-scene-dmrs-losso-seed20261022.json),
[capacity LOSO](results/mvp-capacity-losso-20261022.json),
[profile holdout](results/profile-holdout-baselines-seed20261021.json),
[multipath order](results/multipath-order-stress-seed20261022.json), and
[CAEZ forecast](results/caez-indoor-magnitude-forecast.json).

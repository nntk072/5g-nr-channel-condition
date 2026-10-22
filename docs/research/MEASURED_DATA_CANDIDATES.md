# Measured-data candidates for the 5G NR channel-state plan

**Status:** CAEZ-5G license accepted by the user on 2026-10-22. Official preprocessed indoor, outdoor, and hallway magnitude archives are downloaded locally, passed ZIP CRC/schema checks, and were evaluated in a frozen cross-environment forecast pilot. The archives remain Git-ignored and must not be redistributed. This is measured magnitude-forecast evidence, not LOS/NLOS validation.

## CAEZ-5G (ETH Zurich)

The CAEZ source describes four real wideband, multi-antenna, multi-O-RU 5G NR uplink CSI datasets collected from PUSCH on a 100 MHz n78 testbed centered at 3.45 GHz. The indoor set reports 338,981 samples over 1 h 47 min with synchronized WorldViz UE-position tracking; outdoor and hallway sets provide additional environments and trajectories. The official preprocessing code documents complex per-slot channel estimates (`ch_est`), per-O-RU noise estimates, 273 PRBs, and three DMRS symbols. See the [ETH Zurich CAEZ dataset page](https://iip.ethz.ch/datasets/caez.html) and the [official preprocessing implementation](https://github.com/IIP-Group/neural-positioning/blob/main/gen_dataset.py).

This is a plausible external source for **measured CSI reconstruction and temporal/trajectory channel-state prediction** (T16/T17), subject to inspecting the actual fields and timestamp alignment. It is not a labeled LOS/NLOS dataset: position labels alone do not establish whether each UE–O-RU link is physically LOS. We must not turn its positions into LOS labels without independently verified geometry and obstruction evidence. A measured-channel experiment should freeze the model before test evaluation, split by complete trajectories or contiguous time blocks, and retain noise estimates and complex CSI where available. Random sample splits would leak highly correlated neighboring PUSCH slots.

The CAEZ dataset license permits processing and derivative works with attribution, but prohibits redistribution of the dataset and any subset; it places use at the user's risk. The user explicitly accepted the license and authorized a local research download. The selected official preprocessed file, `caez_5g_indoor_full.npz` under Git-ignored `data/external/caez/`, is 5,957,915,148 bytes and contains downsampled OFDM-domain absolute CSI features. It is substantially smaller than the 394 GB raw indoor archive, but it does not preserve complex phase. The complete file passes ZIP CRC validation and has SHA256 `706e710cc30a96bbe4386e705d9361497e9c2e5df13c00fba77d9318e915c2cc`; its 338,980 rows have finite values and aligned shapes `H[N,4,4,273]` (`float32`), `noise_var[N,4]`, `timestamps[N]`, and `pos[N,2]`. A separate 500-sample official schema subset was audited and then removed; it confirmed the same layout, ancillary arrays, and an object-valued O-RU position dictionary. The pickle-free full audit is recorded in [`caez-indoor-audit.json`](results/caez-indoor-audit.json).

The full-tensor temporal baseline compares persistence with linear-trend extrapolation at a one-second horizon on the chronological final 20%, using 500 ms history. Across 67,717 indoor forecast pairs, macro normalized RMSE is 0.7082 for persistence and 1.4782 for linear trend; persistence wins on all four O-RUs. A separate compact regression predicts each O-RU/Rx antenna's subcarrier-mean magnitude in dB. On the same indoor tail, fixed StandardScaler+Ridge (alpha=1) yields normalized RMSE 0.807 versus 1.047 for persistence and 2.577 for linear trend. Ridge improves this compact summary target in this one indoor campaign, but does not predict subcarrier-selective detail or phase. The aggregate result is [caez-indoor-magnitude-forecast.json](results/caez-indoor-magnitude-forecast.json).

A frozen zero-shot source-to-target pilot is complete. StandardScaler+Ridge (alpha=1) is fit only on the indoor campaign's first 70% of forecast targets; the next 10% is unused, and the final 20% of each campaign is evaluated without target-domain fitting. At a one-second horizon, macro RMSE normalized by each held-out target's standard deviation is:

- Indoor: Ridge 0.807, persistence 1.047, linear trend 2.577. Ridge beats persistence; paired 10-second time-block bootstrap CI for Ridge minus persistence is [-0.258, -0.222].
- Outdoor: Ridge 0.800, persistence 0.915, linear trend 2.035. Ridge beats persistence; paired CI is [-0.131, -0.102].
- Hallway: Ridge 0.945, persistence 0.275, linear trend 0.568. Ridge loses to both baselines; paired CI for Ridge minus persistence is [+0.445, +0.618].

The intervals quantify within-campaign time-block variation, not uncertainty across sites or independent trajectories. This is heterogeneous transfer evidence, not a general model win: only one chronological tail per campaign was tested, and environment is confounded with UE and collection protocol. The model predicts a compact mean-magnitude summary, not the full tensor's subcarrier detail or phase. The separate full-tensor indoor comparison still favors persistence over linear trend.

All three local archives passed ZIP CRC and schema checks; timestamps are strictly increasing and have a median interval of about 20 ms. Their sample counts differ from the current ETH page by -1 indoor (338,980 vs 338,981), -17 outdoor (303,172 vs 303,189), and +2,000 hallway (290,180 vs 288,180). The count discrepancies are unresolved and must be reconciled against the exact official file/version before publication. Maximum observed timestamp gaps are 0.04 s indoor, 0.14 s outdoor, and 5.515 s hallway; alignment uses timestamps and a 20 ms tolerance, so these gaps reduce available pairs rather than being interpolated away. The report records archive hashes, code hashes, split, metrics, and uncertainty: [caez-cross-environment-summary-forecast.json](results/caez-cross-environment-summary-forecast.json). Reproduce with:

    uv run --no-sync --locked python scripts/evaluate_caez_cross_environment.py --train-archive data/external/caez/caez_5g_indoor_full.npz --test-archive data/external/caez/caez_5g_indoor_full.npz data/external/caez/caez_5g_outdoor_full.npz data/external/caez/caez_5g_hallway_full.npz --output docs/research/results/caez-cross-environment-summary-forecast.json

The evaluator allows target-domain observed current and 500 ms lookback summaries as common inputs to Ridge, persistence, and trend; it uses no target labels or target-domain fitting. CAEZ provides no geometric LOS/NLOS labels. The official [ETH Zurich dataset page](https://iip.ethz.ch/datasets/caez.html) reports the campaign counts and different UE/tracking setups; the [preprocessed feature directory](https://iis-people.ee.ethz.ch/~iisdatasets/iip/caez/5g/csi_features/downsampled_ofdm_domain_absolutes/) is recorded for file provenance. Keep archives and any subsets out of Git, public artifacts, and third-party sharing. Cite the CAEZ authors' 2025 Asilomar paper in any public output using these data.

Re-audit the local archive and reproduce the aggregate baseline and compact regression with:

```bash
uv run --no-sync --locked python scripts/audit_caez_npz.py data/external/caez/caez_5g_indoor_full.npz
uv run --no-sync --locked python scripts/evaluate_caez_magnitude_forecast.py data/external/caez/caez_5g_indoor_full.npz --output docs/research/results/caez-indoor-magnitude-forecast.json
```

The forecast command rechecks archive integrity and schema before scoring, so it is safe to run independently. The output contains aggregate metrics and provenance, never raw samples.

## Coverage@CS-KAU

The [Zenodo record](https://zenodo.org/records/22681162) describes indoor 5G NR measurements across 163 points in office, laboratory, and open-space areas, with RSRP, RSRQ, RSSI, and SINR indicators from SSB measurements. Its measurement maps may support measured link-quality studies, but these KPI records are not wideband complex CSI or path-resolved labels. They cannot validate channel reconstruction, delay spread, or geometric LOS classification without further suitable ground truth.

## Admission rule

Before either source is used, record the version/DOI, license, source archive hashes, collection and measurement metadata, field/schema audit, split unit, label definition, and a non-redistributable local storage location. Treat measured-link quality, measured complex CSI, and geometric LOS as different targets. A source is not admitted merely because it is called a 5G channel dataset.

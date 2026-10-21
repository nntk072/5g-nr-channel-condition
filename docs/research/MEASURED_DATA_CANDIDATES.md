# Measured-data candidates for the 5G NR channel-state plan

**Status:** CAEZ-5G license accepted by the user on 2026-10-22; the official preprocessed indoor file is downloaded locally, CRC/SHA/schema-audited, and used in a bounded temporal magnitude-forecast pilot. It remains excluded from Git and redistribution. The result is adjacent measured-channel evidence, not LOS/NLOS validation.

## CAEZ-5G (ETH Zurich)

The CAEZ source describes four real wideband, multi-antenna, multi-O-RU 5G NR uplink CSI datasets collected from PUSCH on a 100 MHz n78 testbed centered at 3.45 GHz. The indoor set reports 338,981 samples over 1 h 47 min with synchronized WorldViz UE-position tracking; outdoor and hallway sets provide additional environments and trajectories. The official preprocessing code documents complex per-slot channel estimates (`ch_est`), per-O-RU noise estimates, 273 PRBs, and three DMRS symbols. See the [ETH Zurich CAEZ dataset page](https://iip.ethz.ch/datasets/caez.html) and the [official preprocessing implementation](https://github.com/IIP-Group/neural-positioning/blob/main/gen_dataset.py).

This is a plausible external source for **measured CSI reconstruction and temporal/trajectory channel-state prediction** (T16/T17), subject to inspecting the actual fields and timestamp alignment. It is not a labeled LOS/NLOS dataset: position labels alone do not establish whether each UE–O-RU link is physically LOS. We must not turn its positions into LOS labels without independently verified geometry and obstruction evidence. A measured-channel experiment should freeze the model before test evaluation, split by complete trajectories or contiguous time blocks, and retain noise estimates and complex CSI where available. Random sample splits would leak highly correlated neighboring PUSCH slots.

The CAEZ dataset license permits processing and derivative works with attribution, but prohibits redistribution of the dataset and any subset; it places use at the user's risk. The user explicitly accepted the license and authorized a local research download. The selected official preprocessed file, `caez_5g_indoor_full.npz` under Git-ignored `data/external/caez/`, is 5,957,915,148 bytes and contains downsampled OFDM-domain absolute CSI features. It is substantially smaller than the 394 GB raw indoor archive, but it does not preserve complex phase. The complete file passes ZIP CRC validation and has SHA256 `706e710cc30a96bbe4386e705d9361497e9c2e5df13c00fba77d9318e915c2cc`; its 338,980 rows have finite values and aligned shapes `H[N,4,4,273]` (`float32`), `noise_var[N,4]`, `timestamps[N]`, and `pos[N,2]`. A separate 500-sample official schema subset was audited and then removed; it confirmed the same layout, ancillary arrays, and an object-valued O-RU position dictionary. The pickle-free full audit is recorded in [`caez-indoor-audit.json`](results/caez-indoor-audit.json).

A reproducible temporal magnitude-forecast baseline compares persistence with linear-trend extrapolation at a one-second horizon using the chronological final 20% and a 500 ms history. Across 67,717 forecast pairs, scene-macro normalized RMSE is `0.7082` for persistence and `1.4782` for linear trend; persistence wins on all four O-RUs. This is a simple, single-indoor-campaign pilot with no learned predictor, no cross-environment test, and no LOS/NLOS labels. The aggregate result is [`caez-indoor-magnitude-forecast.json`](results/caez-indoor-magnitude-forecast.json). It supports only a measured CSI-magnitude forecasting observation, not geometric LOS classification or phase-sensitive reconstruction. Keep the archive and any subset/derived data out of Git, public artifacts, and third-party sharing. Cite the CAEZ authors' 2025 Asilomar paper in any public output using these data. The official [license](https://iip.ethz.ch/datasets/caez/caez-dataset-license.html) and [preprocessed feature directory](https://iis-people.ee.ethz.ch/~iisdatasets/iip/caez/5g/csi_features/downsampled_ofdm_domain_absolutes/) are recorded for provenance.

Once the full local transfer completes, audit and run the aggregate baseline with:

```bash
uv run --no-sync --locked python scripts/audit_caez_npz.py data/external/caez/caez_5g_indoor_full.npz
uv run --no-sync --locked python scripts/evaluate_caez_magnitude_forecast.py data/external/caez/caez_5g_indoor_full.npz --output docs/research/results/caez-indoor-magnitude-forecast.json
```

The forecast command rechecks archive integrity and schema before scoring, so it is safe to run independently. The output contains aggregate metrics and provenance, never raw samples.

## Coverage@CS-KAU

The [Zenodo record](https://zenodo.org/records/22681162) describes indoor 5G NR measurements across 163 points in office, laboratory, and open-space areas, with RSRP, RSRQ, RSSI, and SINR indicators from SSB measurements. Its measurement maps may support measured link-quality studies, but these KPI records are not wideband complex CSI or path-resolved labels. They cannot validate channel reconstruction, delay spread, or geometric LOS classification without further suitable ground truth.

## Admission rule

Before either source is used, record the version/DOI, license, source archive hashes, collection and measurement metadata, field/schema audit, split unit, label definition, and a non-redistributable local storage location. Treat measured-link quality, measured complex CSI, and geometric LOS as different targets. A source is not admitted merely because it is called a 5G channel dataset.

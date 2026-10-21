# Measured-data candidates for the 5G NR channel-state plan

**Status:** source-screened; no external dataset has been downloaded or admitted to an experiment yet.

## CAEZ-5G (ETH Zurich)

The CAEZ source describes four real wideband, multi-antenna, multi-O-RU 5G NR uplink CSI datasets collected from PUSCH on a 100 MHz n78 testbed centered at 3.45 GHz. The indoor set reports 338,981 samples over 1 h 47 min with synchronized WorldViz UE-position tracking; outdoor and hallway sets provide additional environments and trajectories. The official preprocessing code documents complex per-slot channel estimates (`ch_est`), per-O-RU noise estimates, 273 PRBs, and three DMRS symbols. See the [ETH Zurich CAEZ dataset page](https://iip.ethz.ch/datasets/caez.html) and the [official preprocessing implementation](https://github.com/IIP-Group/neural-positioning/blob/main/gen_dataset.py).

This is a plausible external source for **measured CSI reconstruction and temporal/trajectory channel-state prediction** (T16/T17), subject to inspecting the actual fields and timestamp alignment. It is not a labeled LOS/NLOS dataset: position labels alone do not establish whether each UE–O-RU link is physically LOS. We must not turn its positions into LOS labels without independently verified geometry and obstruction evidence. A measured-channel experiment should freeze the model before test evaluation, split by complete trajectories or contiguous time blocks, and retain noise estimates and complex CSI where available. Random sample splits would leak highly correlated neighboring PUSCH slots.

The CAEZ dataset license permits processing and derivative works with attribution, but prohibits redistribution of the dataset and any subset; it places use at the user's risk. Any downloaded source must remain outside Git and public artifacts. The user has been asked to confirm acceptance before download; until then, the measured-data gate remains open. Cite the CAEZ authors' 2025 Asilomar paper in any public output using these data.

## Coverage@CS-KAU

The [Zenodo record](https://zenodo.org/records/22681162) describes indoor 5G NR measurements across 163 points in office, laboratory, and open-space areas, with RSRP, RSRQ, RSSI, and SINR indicators from SSB measurements. Its measurement maps may support measured link-quality studies, but these KPI records are not wideband complex CSI or path-resolved labels. They cannot validate channel reconstruction, delay spread, or geometric LOS classification without further suitable ground truth.

## Admission rule

Before either source is used, record the version/DOI, license, source archive hashes, collection and measurement metadata, field/schema audit, split unit, label definition, and a non-redistributable local storage location. Treat measured-link quality, measured complex CSI, and geometric LOS as different targets. A source is not admitted merely because it is called a 5G channel dataset.

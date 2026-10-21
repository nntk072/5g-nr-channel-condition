# Three distinct, deterministic research simulation geometries

`city41`, `city42`, `city43` have seeds 41/42/43, a 4x4 building grid,
material variation and eight simplified vehicle occluders. They are **synthetic**;
no real-world geometric fidelity or ray-tracing execution is claimed.

In these 4x4 scenes, a suitable starting street-intersection setup is
`--tx 0 -55 18 --center 0 0 1.5 --radius 115`. Tune this per scene based
on visual inspection and LOS/NLOS balanced sampling. For the 5x5 geometries
created in Quickstart, use the explicit positions shown there instead.

The scene generator saves a companion `*.manifest.json` (object bounds,
materials, config/mesh hashes). Sionna RT generation uses these bounds to
avoid placing receivers inside modeled solid obstacles.

For strong academic work, evaluate additional reconstructed and measured
geometries with correctly calibrated radio materials and antennas.

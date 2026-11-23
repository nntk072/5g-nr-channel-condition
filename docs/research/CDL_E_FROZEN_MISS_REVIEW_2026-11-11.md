# CDL-E four-element covariance frozen-miss review

**Disposition:** source review found no arithmetic defect; the frozen native
result remains a statistical failure with cause unresolved. This review does
not change the protocol, thresholds, result bytes, or G4 qualification status.

## Frozen evidence

The retained run is
`docs/research/results/g4-cdl-four-element-covariance-frozen-execution-20261109.json`:
seed 6714, 8,192 channel draws per CDL profile, 16 unique real covariance
coordinates per profile, two-sided 99% Bonferroni intervals, and maximum
half-width 0.05. CDL-A–D pass. CDL-E fails only `Re(C[0,1])`: absolute error
0.0135743013931, half-width 0.0129694115595, excess 0.0006048898336.

The implied standard error is `half_width / z*`, with
`z* = NormalDist.inv_cdf(1 - 0.01/(2*16)) = 3.4205267011`. The frozen error is
therefore 3.580059 standard errors from the reference. This is a narrow miss,
but its preregistered decision remains **FAIL**. The later 65,536-draw
diagnostic has z=0.515 and was selected after the miss; it is not evidence that
changes this decision.

## Independent source derivation review

The reviewed implementation is
`tests/test_integrations/test_g4_cdl_spatial_covariance_4element_native.py` at
SHA-256
`71193b36a4bcafd9269561c1da5ba2143b0161fad8f1f2fd4560402e26950295`. Its
fixture `docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json` is pinned at
SHA-256
`24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`; that
fixture records the ETSI source digest
`b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.

- **Power normalization:** component powers are converted from dB to linear
  power and normalized once across the LOS specular component and all diffuse
  clusters. For CDL-D/E, the fixture and assertion require the specular row
  first, followed by every diffuse row. Diffuse cluster weights select exactly
  those latter rows; the LOS contribution uses only the specular weight. Their
  weights sum to one, so each covariance diagonal is one.
- **Ray-angle coupling:** each cluster forms the Cartesian product of the 20
  azimuth offsets and 20 zenith offsets. This computes the expectation over
  independent azimuth/zenith coupling, rather than pairing equal ray indices.
  Cluster power is applied after each cluster's 400-pair average.
- **Array phase:** for the local-y, half-wavelength ULA at zero orientation,
  the direction projection is `sin(ZoA) * sin(AoA)`. The oracle uses
  `exp(-j*pi*projection*(column-row))`, matching the observed
  `h[row] * conj(h[column])` convention. Reversing the convention conjugates
  the covariance and cannot by itself explain a real-part miss.
- **LOS and diffuse composition:** CDL-D/E specular steering is added as its
  own normalized component; the diffuse first cluster remains in the diffuse
  sum even though its nominal delay and cluster angles coincide with LOS. This
  follows the component split retained in the fixture and the pinned CDL
  table's LOS K-factor handling. The native comparison sums all path taps into
  one narrowband channel vector per draw, so it includes both contributions.
- **Claim extraction:** the test forms one outer product per draw as
  `h[:, :, None] * h[:, None, :].conj()`, then extracts four diagonal powers
  and real/imaginary parts of the six unique upper-triangle entries. It asserts
  16 coordinates, computes draw-level standard errors with `ddof=1`, and
  applies the declared Bonferroni normal quantile. The acceptance comparison
  and precision ceiling use the same normalized-power scale.

The five dependency-light reference checks pass. They establish the reference
matrix is finite, Hermitian, unit-diagonal and positive semidefinite; they do
not settle the stochastic native interval failure. The reviewed source and
retained claim do not establish whether the one-coordinate miss is ordinary
sampling variation or an unrecorded simulator/reference mismatch. No specific
code defect is justified by this evidence.

## Later preregistered replication

Before any new native run, prepare a separate protocol that:

1. reserves a fresh independent seed roster after checking all currently
   retained G4 seed rosters; neither seed 6714 nor diagnostic seed 3000000 may
   be reused;
2. freezes the exact CDL-E array, orientation, carrier, Sionna version, fixture
   and test-source digests before execution;
3. keeps the 16-coordinate, 99% family-wise interval and 0.05 half-width
   ceiling unchanged, with the draw count and seed roster fixed before outcomes
   are observed;
4. retains every coordinate estimate, standard error, interval, draw count and
   seed identity so an independent auditor can recompute the decision; and
5. preserves both the original failure and the new outcome. A later pass may
   show the failure was not repeatable under the frozen setup, but it cannot
   retroactively turn the 2026-11-09 result into a pass.

Do not launch that replication as part of this source review. G4 and production
RT remain gated on the governing plan's remaining source, statistical and
independent-evidence requirements.

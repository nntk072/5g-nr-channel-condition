# Profile-likelihood delay-set stress protocol

`experiments/identifiability_profile_likelihood.py` fits an unordered two-path
complex CFR model and reports both profile intervals and split-conformal
sublevel sets. The stress controls add an assumed/true noise-variance ratio, an
optional weaker third specular path, and random permutations of physical path
arrays. The estimand in three-path cases is fixed in advance as the delay
difference between the strongest and second-strongest paths; the fitted model
still contains two paths and is intentionally misspecified.

For one fixed, declared synthetic condition, calibration and evaluation are
independent channel realizations from the same condition. If calibration
nonconformity scores are `s_1, ..., s_m`, the threshold is the order statistic
at rank `ceil((m+1)(1-alpha))`; if that rank exceeds `m`, the implementation
returns an unbounded threshold. The reported set contains delay-grid points
whose score does not exceed the calibration threshold. Under exchangeability,
the rank argument gives marginal finite-sample coverage over both calibration
and future observations. Ties make this rule conservative. It does not provide
conditional coverage for a fixed calibration sample, nor transfer across
conditions unless those conditions are exchangeable with the calibration data.

The committed stress summary is compact and binds the ignored full report by
SHA256. It retains condition, calibration rank and threshold, empirical
coverage, a Wilson interval conditional on that fitted threshold, and median
set span. That Wilson interval does not include calibration-threshold
variability. A single seeded stress run is a diagnostic; it cannot validate
calibration on measured channels, arbitrary multipath, changing receiver noise,
or unseen Sionna RT scenes. Repeat-calibration experiments and independent
physical evidence remain required before making a calibration claim beyond the
stated synthetic conditions.

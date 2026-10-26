# G5 TDL-A–E DM-RS density family replication protocol, revision 2

**Status:** preregistered before the revision-2 run. Revision 1 was invalidated
because its family wrapper counted report metadata as acceptance claims and
therefore emitted a family `NOT_MET` despite all five nested profile reports
passing. The retained revision-1 artifact is preserved as a software-defect
diagnostic only; its seeds and outcomes are not reused, pooled, or promoted to
confirmatory evidence. The runner fix is limited to selecting the three named
claim booleans explicitly. Revision 2 uses a wholly disjoint seed range.

## Question and frozen design

The question and scientific criteria are unchanged from revision 1. Run 80
independent channel draws per profile using seeds 936001–936080 for each of
TDL-A, TDL-B, TDL-C, TDL-D, and TDL-E. These seeds are disjoint from the
original 64-seed family (934001–934064) and invalid revision-1 run
(935001–935080). No outcomes are pooled.

Use 100-ns delay spread, 3.5-GHz carrier, 120-km/h mobile speed, 30-kHz SCS,
12-RB PUSCH, 18-dB configured receive-grid SNR, and static-position-0,
mobile-position-0, and mobile-position-3 conditions. Run on CPU with CP
overrun disabled and complete CP coverage required. Pair payload/noise for
static and mobile position 0, and channel truth/noise for mobile positions 0
and 3. A channel seed is the independent unit; its transport block is nested.

Keep family-wise alpha at 0.01 over 15 claims, with Bonferroni alpha 0.01/15,
20,000 paired channel-seed bootstrap replicates, recorded profile-specific
bootstrap seeds, and one-sided exact Clopper–Pearson bounds for static
failure prevalence. Each profile must pass all three explicit acceptance
claims: static BLER ≤0.05 and adjusted exact upper bound ≤0.10; mobile
position-0 minus position-3 BLER reduction ≥0.10 with adjusted interval lower
bound >0; and adjusted interval lower bound >0 for position-3 minus
position-0 decoded bits per slot. Any failed or missing claim makes the
family `NOT_MET`.

No parameter, seed, configuration, or endpoint may be changed after this
run starts. Any new implementation defect requires another revision and
another disjoint seed set. A pass supports only these five named profiles at
this fixed configuration; it is not measured-channel validity, broad G5.09
closure, or production RT authorization.

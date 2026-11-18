# Multi-link RT to network PHY support

This matrix defines the software adapter boundary from stored UMCD physical
channel views to `integrations.sionna.network_phy.compose_downlink_resource_grids`.
It describes input support, not native solver qualification or a 3GPP receiver
claim.

| Dimension | Supported by the adapter | Explicitly rejected or outside this contract |
|---|---|---|
| Nodes | One or more GNB transmitters and UE receivers; `INTERFERER` nodes may transmit interference only | Uplink, sidelink, non-TX/RX roles, or a non-GNB serving node |
| Directed links | Exactly one downlink link for every TX×RX pair | Missing or duplicate pair links, non-downlink directions |
| Link role | Exactly one serving transmitter per receiver; every other pair is `interfering` | Zero/multiple serving transmitters or `OTHER`/`SOUNDING` pair roles |
| Array ports | Equal port count across all TX nodes and separately across all RX nodes; validated per-link port maps may reorder or select ports | Heterogeneous TX or RX port counts |
| Channel views | Exactly one valid `PhysicalChannelView` per link, absolute propagation delay, absolute power, exact episode frame/carrier/subcarrier identity | Missing/duplicate views, masked invalid channel elements, relative-delay or normalized-power views |
| Time/frequency | Each stored per-link frequency response is repeated over the requested OFDM symbols; episode frames become the network batch axis | Per-symbol Doppler evolution, cross-numerology composition, or implicit interpolation |
| Failure/no path | Invalid masks and incomplete topology fail closed | Missing link/channel data are never filled as measured zero channels |
| PHY model | Synchronized, aligned frequency-domain superposition followed by serving/interference SINR measurement | Relative timing/CFO, adjacent-channel leakage, RF nonlinearity, or receiver qualification |

`SionnaSceneNetwork` extraction preserves device-axis indices and port maps.
For mixed serving/interference networks, pass an explicit
`link_roles_by_association` entry for every selected TX→RX association; a
single shared `link_role` remains available for legacy homogeneous layouts.
The extractor rejects missing or extra role assignments, while the network
PHY adapter validates that each UE has exactly one GNB serving link.
`network_episode_identity()` hashes the complete ordered topology, including
array weights, node roles/poses, link roles/port maps, grids, split relations,
and pose/time-alignment digests. `build_rt_network_generation_request()` uses
that identity in the D1 request and rejects incomplete scene assets or
topologies outside the supported downlink matrix. It only prepares the exact
request for authorization. `run_authorized_static_network_episode()` consumes
that request only after D1 authorization and destination checks, then verifies
the live scene asset closure, material state, geometry inventory, carrier,
topology, and run-identity binding before it creates a solver. It publishes a
topology-bound bundle only for a successful solve. Candidate-cap saturation
and solver exceptions instead produce masked path/channel layers and an
immutable digest-bound receipt with explicit `UNCONVERGED` or
`SOLVER_FAILURE` status. This runner is a software integration; its deterministic
fake-solver tests do not qualify native Sionna behavior or physical channel
truth.

Use `episode_channels_to_network_phy_input()` after reading a complete
`EpisodeBundle`. It returns node order, frame/grid identity, serving-TX indices,
and a seven-axis complex tensor with axes
`[frame, rx, rx_port, tx, tx_port, symbol, subcarrier]`. Port cells excluded by
an explicit link-local port map are structural zeros; missing directed links
are errors. The returned `episode_id` keeps the tensor tied to the stored
episode.

For a static solve snapshot at each declared episode frame,
`sionna_bindings_to_channel_views()` converts the complete link/frame binding
set directly from `Paths.cfr()` into one absolute CFR view per episode link. It
uses the episode frequency offsets and link-local port maps, with delay and
power normalization disabled. Failed or unconverged solves produce masked
channel elements with zero storage; the network PHY rejects those views. A
successful empty ray result is retained as a valid zero under the finite
solver budget, which is still not evidence of physical path absence. This
producer adapter does not generate a Sionna scene, run `PathSolver`, or claim
dynamic within-frame channel evolution.

`build_sionna_rt_episode_bundle_from_paths()` composes this conversion with the
canonical path/provenance bundle builder, so the generated path and channel
layers share one episode and run identity. The caller can pass the resulting
bundle to the immutable episode-store writer, or use
`write_sionna_rt_episode_bundle_from_paths()` to apply the existing D1
destination gate and publish it in one call. D1 authorization remains
mandatory for publication; this helper does not bypass any scientific gate.

The analytical contract test writes a synthetic 2×2 episode and its four
channel views, reads the immutable bundle, composes the stored channels with
per-TX grids, and measures serving and interference power through the existing
network PHY. It does not run `PathSolver`, establish channel truth, or qualify a
receiver. The separate native 2×2 Sionna path-binding test remains a narrow API
check, not a qualification result. Dependency-light producer tests verify
device/port selection, frame coverage, absolute normalization arguments, and
failure masking.

# G4 scenario GBSM generation command v1

The Sionna 2.2 scenario adapter can be invoked through the supported CLI. It writes a content-addressed scenario run store and one canonical `EpisodeBundle` per explicitly named drop:

~~~sh
loslab scenario-gbsm-data --config scenario-umi.json --output-dir data/scenario-gbsm
~~~

The config uses schema umcd.scenario_gbsm_generation.v1, defined in datasets/schema/scenario-gbsm-generation-config.v1.schema.json. It must name a unique split-parent ID for every generated drop and give explicit UT/BS location, UT velocity, indoor state, and scenario LOS state. Optional endpoint orientations and array settings are accepted. All scenario-specific Sionna options must be declared in generator.scenario_options.

The CLI rejects batches over 128 drops, grids over 4,096 subcarriers, time grids over 256 samples, products over two million batch × subcarrier × time elements, and arrays over 256 ports per endpoint before generation. The underlying adapter validates scenario-specific option allowlists and tensor axes. Stored records retain the scenario LOS state separately and leave geometric LOS unknown. The v2 generation report lists both the run-store manifest and each episode-bundle manifest. Each episode bundle binds the drop ID as its split subject, binds its split group to the generating run, records the generated array layout and static orientations, and carries a pose timeline for multi-time-step UT motion. Strict child-level release audits should roster these per-drop EpisodeBundles; the batch run store is a generation-level companion. Site-level independence across runs remains unestablished.

Outputs are labeled as unqualified standardized-model simulations. The bundle contains no physical RT path oracle; stochastic GBSM paths remain a distinct channel layer. The command does not claim site or drop independence, qualify scenario statistics, or authorize production RT generation or advanced training.

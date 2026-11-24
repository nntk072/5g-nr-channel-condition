# datasets

**Purpose.** NR channel records and the contracts that bind them: channel
datasets, multi-node episodes, stored episode bundles, path-oracle tables,
targets, split policies with leakage relations, lineage claims, and the JSON
schemas in `datasets/schema/`.

**Public API (entry points).**

- Channel records: `datasets.channel` (`ChannelDataset`), `datasets.channel_view`
  (`PhysicalChannelView`), `datasets.standardized` (TDL/CDL records).
- Episodes: `datasets.network` (`NetworkEpisode`), `datasets.episode_store`
  (`write_episode_bundle`, `read_episode_bundle`).
- Oracles and targets: `datasets.path_oracle` (`PathOracleTable`),
  `datasets.targets`, `datasets.counterfactual`.
- Splits: `datasets.split_policy` (signed policies, registries, leakage
  relations), `datasets.split_lineage`, `datasets.splits`,
  `datasets.independent_splits`.
- Provenance and evidence: `datasets.provenance`, `datasets.lineage`,
  `datasets.public_release`.

**Invariants.**

- Oracle truth (geometric line-of-sight, path-oracle delays and gains) is stored
  separately from model inputs. `datasets.model_input_audit` rejects any oracle
  field that crosses the model-input boundary.
- Records are immutable after validation. Storage is pickle-free.
- Split membership is decided by scene-grouped identities before any model sees
  data. Signed split locks are never regenerated silently.
- Schema identity strings (`$id`, `schema` constants) are versioned. A change to a
  schema's meaning requires a new version.

# Expected-population and outcome audit

Selection and missingness rates cannot be recovered from a dataset containing
only accepted rows. Freeze the population roster before generation or filtering,
with one unique `unit_id` per expected dataset record and the expected family,
domain, independent-group identity, disposition, and reason. Supported
dispositions are `included`, `excluded`, `rejected`, `no_path`, and `empty`.
Only `included` rows belong in the ChannelDataset; every included row must carry
that roster `unit_id` as `metadata.sample_id`. Other outcomes require an explicit
reason. This keeps solver noncoverage distinct from a dropped or silently
excluded observation.

The versioned input contract is
`datasets/schema/expected-population-roster.v1.schema.json`. To audit one or more
ChannelDataset files against it, run:

```sh
loslab-audit-channel-population \
  --roster frozen-population.json \
  --input accepted-part-000.npz accepted-part-001.npz \
  --output population-audit.json
```

The command binds the exact roster bytes and each dataset file digest, checks
sample/family/domain/group joins, and reports inclusion, omission, explicit
exclusion/rejection, no-path, and empty counts by family and domain. It writes a
new report without replacement. A roster mismatch still produces a report and
returns exit status 2. `ROSTER_MATCH` means only that observed rows agree with
the declared roster; it does not prove the roster is complete for the target
population, disposition reasons are true, or group identities are physically
independent. Those claims require authoritative inventory and independent
review outside this software audit.

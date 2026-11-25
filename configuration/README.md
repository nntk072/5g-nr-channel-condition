# configuration

**Purpose.** Load YAML and JSON run configurations and attach a stable identity
to each resolved configuration, so that every run records exactly which settings
produced it.

**Public API.**

- `configuration.loader` — loads a configuration file and returns the resolved
  mapping together with its canonical identity (SHA-256 over canonical JSON).

**Invariants.**

- A configuration's identity depends only on its resolved content, not on file
  name, key order, or whitespace.
- Unknown or duplicate keys are rejected rather than silently overridden.
- Loading performs no network access and does not execute code embedded in
  configuration files.
- This package depends on nothing else in the repository.

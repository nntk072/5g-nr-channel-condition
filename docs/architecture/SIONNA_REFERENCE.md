# External Sionna source reference

The upstream Sionna repository is cloned outside this thesis repository:

```text
/home/nguyenl37/sionna
```

At the time this refactor was prepared, the checkout was on `main` at commit
`2f0a3820269b53fd022780a6f2f5f720f7ceb9b4`, from
`https://github.com/NVlabs/sionna.git`.

Use this checkout to inspect upstream implementation and API details. It is not
a submodule, vendored dependency, or Python import path for this thesis. Runtime
experiments use the separately installed Sionna package; record its exact
version in each run manifest.

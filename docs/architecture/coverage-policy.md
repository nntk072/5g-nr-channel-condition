# Coverage policy

The CI gate requires **100 % branch coverage** of the product packages
(`[tool.coverage.run] source` in `pyproject.toml`).

Exclusions are the only way a line or branch may be left out of the gate. Each
one is listed below with its reason. `tools/coverage_policy.py` runs in the
architecture gate and fails when a configured exclusion has no row here, or when
a row no longer matches the configuration.

Each row is a single table line: the first column is the exact configured entry
in code formatting, the second column is the reason.

## Exclusions

| Entry | Reason |
| --- | --- |
| `if TYPE_CHECKING:` | Import-time branch for type checkers only; it never runs. |
| `if __name__ == .__main__.:` | Module entry-point guard. The behaviour is tested through the callable `main()`. |

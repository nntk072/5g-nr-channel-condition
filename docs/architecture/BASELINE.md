# Refactor baseline

Before moving the research modules, the existing `tests_research/` suite was
run from the repository root with the project's declared research dependencies
provided by `uv`:

```bash
uv run --with pytest --with 'numpy>=1.26,<3' \
  --with 'scipy>=1.12' --with 'scikit-learn>=1.5,<2' \
  python -m pytest -c pytest-research.ini -q
```

Result: **47 passed**. The initial attempt to invoke the `pytest` console script
did not put the repository root on `sys.path`; the baseline command above uses
`python -m pytest` and passed.

After the first package migration and architecture checks were introduced, the
suite reported **50 passed**. Additional artifact/configuration tests and the
final model/learning split were added afterward and require a fresh full test
pass before this branch is considered validated.

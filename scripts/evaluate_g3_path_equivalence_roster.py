"""Script entry point for the packaged G3 path-roster evaluator."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.g3_path_equivalence_roster_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

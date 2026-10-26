"""Check complete OpenSSH-signature coverage for a G3.03 proposal package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.g3_signed_attestation import verify_g3_signed_attestation_set  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "attestations",
        nargs="+",
        type=Path,
        help="canonical attestation JSON sidecars; each must have an adjacent .sig file",
    )
    parser.add_argument("--allowed-signers", required=True, type=Path)
    parser.add_argument("--signer-policy", required=True, type=Path)
    parser.add_argument(
        "--require-complete-qualification-attestations",
        action="store_true",
        help="require every G3 attestation type before independent scientific review",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_g3_signed_attestation_set(
            args.package,
            args.attestations,
            args.allowed_signers,
            args.signer_policy,
            require_complete_qualification_attestations=(
                args.require_complete_qualification_attestations
            ),
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

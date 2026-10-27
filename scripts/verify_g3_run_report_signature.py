"""Verify a detached OpenSSH signature over a saved G3 run-verification report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.g3_joint_evidence_run_manifest import (
    verify_g3_joint_evidence_run_report_signature,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="saved G3 run-verification report JSON")
    parser.add_argument("--signature", type=Path, required=True, help="detached report signature")
    parser.add_argument(
        "--trusted-allowed-signers",
        type=Path,
        required=True,
        help="externally governed OpenSSH allowed_signers file",
    )
    parser.add_argument("--signer-id", required=True, help="expected OpenSSH signer principal")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional receipt destination; must not exist",
    )
    args = parser.parse_args(argv)
    try:
        receipt = verify_g3_joint_evidence_run_report_signature(
            args.report,
            args.signature,
            args.trusted_allowed_signers,
            signer_id=args.signer_id,
        )
        rendered = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
        print(rendered, end="")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())

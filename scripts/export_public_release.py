"""Export a hash-checked public release after external signature verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets.public_release import export_public_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="root containing manifest and artifacts")
    parser.add_argument("manifest", type=Path, help="strict public-release manifest JSON")
    parser.add_argument("archive", type=Path, help="new ZIP archive destination")
    parser.add_argument("--signature", type=Path, required=True, help="detached manifest signature")
    parser.add_argument(
        "--trusted-allowed-signers",
        type=Path,
        required=True,
        help="externally governed OpenSSH allowed_signers file",
    )
    parser.add_argument("--signer-id", required=True, help="expected release reviewer principal")
    args = parser.parse_args(argv)
    try:
        archive_path = export_public_release(
            args.source_root,
            args.manifest,
            args.archive,
            manifest_signature_path=args.signature,
            trusted_allowed_signers_path=args.trusted_allowed_signers,
            signer_id=args.signer_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"archive": str(archive_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

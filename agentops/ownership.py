"""Writable-path ownership checks for ready task cards."""

from __future__ import annotations

from pathlib import PurePosixPath

from .contracts import ContractError


def paths_overlap(left: str, right: str) -> bool:
    a = PurePosixPath(left).parts
    b = PurePosixPath(right).parts
    if not a or not b:
        # An empty path such as "." is the repository root; treat it as overlapping
        # everything so that ownership checks fail closed rather than fail open.
        return True
    common = min(len(a), len(b))
    return a[:common] == b[:common]


def validate_ready_ownership(cards: list[dict]) -> None:
    ready = [card for card in cards if card.get("effective_status", card["status"]) == "ready"]
    for index, left in enumerate(ready):
        for right in ready[index + 1 :]:
            for left_path in left["owned_paths"]:
                for right_path in right["owned_paths"]:
                    if paths_overlap(left_path, right_path):
                        raise ContractError(
                            f"ready tasks {left['task_id']} and {right['task_id']} overlap at "
                            f"{left_path!r} and {right_path!r}"
                        )

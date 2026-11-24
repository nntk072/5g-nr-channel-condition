"""Split-policy lock reads, fraction validation and subject adapters: refusal of unsafe or malformed inputs."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest
from datasets.split_policy import SplitPolicy
from datasets.split_policy import policy as policy_module
from datasets.split_policy.policy import _read_immutable_lock_bytes


def test_lock_reader_returns_exact_bytes_and_enforces_its_limit(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_bytes(b"0123456789")
    assert _read_immutable_lock_bytes(lock, max_bytes=10) == b"0123456789"
    with pytest.raises(ValueError, match="exceeds its 4-byte limit"):
        _read_immutable_lock_bytes(lock, max_bytes=4)
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        _read_immutable_lock_bytes(lock, max_bytes=0)
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        _read_immutable_lock_bytes(lock, max_bytes=True)


def test_lock_reader_refuses_symlink_leaves(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _read_immutable_lock_bytes(link, artifact_label="policy lock")


def test_symlink_swapped_in_after_the_stat_is_refused_by_no_follow_open(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "swapped.json"
    link.symlink_to(target)
    real_lstat = os.lstat

    def regular_looking_lstat(path, *args, **kwargs):
        # Simulate the race: the leaf looked like a file at stat time but is a symlink at open.
        if Path(path) == link:
            return real_lstat(target)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(policy_module.os, "lstat", regular_looking_lstat)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _read_immutable_lock_bytes(link)


def test_unreadable_lock_surfaces_the_operating_system_error(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "unreadable.json"
    lock.write_bytes(b"{}")

    def denied(path, flags, *args, **kwargs):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(policy_module.os, "open", denied)
    with pytest.raises(OSError, match="denied"):
        _read_immutable_lock_bytes(lock)


def test_fractions_must_be_finite_real_and_sum_to_one() -> None:
    policy = SplitPolicy.release_v2(seed="fractions")
    assert policy.to_dict()["seed"] == "fractions"
    with pytest.raises(ValueError, match="three-way or four-way"):
        SplitPolicy(seed="s", split_fractions={"train": 0.5, "test": 0.5})
    with pytest.raises(ValueError, match="finite and positive"):
        SplitPolicy(
            seed="s",
            split_fractions={"train": 10**400, "validation": 0.2, "test": 0.3},
        )
    with pytest.raises(TypeError, match="must be real numbers"):
        SplitPolicy(seed="s", split_fractions={"train": "0.5", "validation": 0.2, "test": 0.3})
    with pytest.raises(ValueError, match="finite and positive"):
        SplitPolicy(seed="s", split_fractions={"train": 0.0, "validation": 0.5, "test": 0.5})
    with pytest.raises(ValueError, match="sum to one"):
        SplitPolicy(seed="s", split_fractions={"train": 0.5, "validation": 0.2, "test": 0.2})


def test_episode_and_draw_adapters_refuse_foreign_objects() -> None:
    with pytest.raises(TypeError, match="episode must be a NetworkEpisode"):
        policy_module.network_episode_subject("not-an-episode")
    with pytest.raises(TypeError, match="record must be a StandardizedChannelRecord"):
        policy_module.standardized_draw_subject("not-a-record")

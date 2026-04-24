#!/usr/bin/env python3
"""Host-side harness for IQ-1351 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_runtime_prefix_cache_replay_guard_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    PREFIX_CACHE_ERR_BAD_PARAM,
    PREFIX_CACHE_ERR_NOT_FOUND,
    PREFIX_CACHE_ERR_NULL_PTR,
    PREFIX_CACHE_ERR_POLICY,
    PREFIX_CACHE_OK,
    PREFIX_CACHE_PROFILE_DEV_LOCAL,
    PREFIX_CACHE_PROFILE_SECURE_LOCAL,
    _cache_with_tick,
)
from test_runtime_prefix_cache_replay_guard_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_runtime_prefix_cache_replay_guard_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    cache,
    entry_index: int,
    access_tick: int,
    profile_id: int,
    outputs: list[int] | None,
    slot_previous: int,
    slot_applied: int,
    slot_guard: int,
):
    if cache is None or cache.entries is None or outputs is None:
        return PREFIX_CACHE_ERR_NULL_PTR
    if slot_previous == slot_applied or slot_previous == slot_guard or slot_applied == slot_guard:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if entry_index < 0 or entry_index >= cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].valid != 1:
        return PREFIX_CACHE_ERR_NOT_FOUND
    if access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if profile_id not in (PREFIX_CACHE_PROFILE_SECURE_LOCAL, PREFIX_CACHE_PROFILE_DEV_LOCAL):
        return PREFIX_CACHE_ERR_BAD_PARAM

    snapshot_entry_index = entry_index
    snapshot_access_tick = access_tick
    snapshot_profile_id = profile_id
    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_previous_tick = cache.entries[entry_index].last_used_tick

    preflight_out = [0x13579BDF, 0x2468ACE0, 0x10203040]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        cache,
        entry_index,
        access_tick,
        profile_id,
        preflight_out,
        0,
        1,
        2,
    )
    if status != PREFIX_CACHE_OK:
        return status

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count != snapshot_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if (
        entry_index != snapshot_entry_index
        or access_tick != snapshot_access_tick
        or profile_id != snapshot_profile_id
    ):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if preflight_out != [0x13579BDF, 0x2468ACE0, 0x10203040]:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM

    cache.entries[entry_index].last_used_tick = snapshot_previous_tick

    commit_out = [0, 0, 0]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        cache,
        entry_index,
        access_tick,
        profile_id,
        commit_out,
        0,
        1,
        2,
    )
    if status != PREFIX_CACHE_OK:
        return status

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count != snapshot_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if (
        entry_index != snapshot_entry_index
        or access_tick != snapshot_access_tick
        or profile_id != snapshot_profile_id
    ):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if commit_out[0] < 0 or commit_out[1] < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if commit_out[2] not in (0, 1):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if commit_out[1] != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != commit_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if commit_out[0] != snapshot_previous_tick or commit_out[1] != access_tick or commit_out[2] != 1:
        return PREFIX_CACHE_ERR_BAD_PARAM

    outputs[slot_previous] = commit_out[0]
    outputs[slot_applied] = commit_out[1]
    outputs[slot_guard] = commit_out[2]
    return PREFIX_CACHE_OK


def test_strict_parity_secure_local_monotonic_roundtrip():
    cache = _cache_with_tick(700)
    out = [-1, -1, -1]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        cache,
        entry_index=1,
        access_tick=701,
        profile_id=PREFIX_CACHE_PROFILE_SECURE_LOCAL,
        outputs=out,
        slot_previous=0,
        slot_applied=1,
        slot_guard=2,
    )
    assert status == PREFIX_CACHE_OK
    assert out == [700, 701, 1]
    assert cache.entries[1].last_used_tick == 701


def test_strict_parity_dev_local_allows_rollback():
    cache = _cache_with_tick(910)
    out = [9, 8, 7]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        cache,
        entry_index=1,
        access_tick=600,
        profile_id=PREFIX_CACHE_PROFILE_DEV_LOCAL,
        outputs=out,
        slot_previous=0,
        slot_applied=1,
        slot_guard=2,
    )
    assert status == PREFIX_CACHE_OK
    assert out == [910, 600, 1]
    assert cache.entries[1].last_used_tick == 600


def test_strict_parity_secure_local_policy_failure_keeps_outputs_unchanged():
    cache = _cache_with_tick(400)
    out = [1, 2, 3]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        cache,
        entry_index=1,
        access_tick=399,
        profile_id=PREFIX_CACHE_PROFILE_SECURE_LOCAL,
        outputs=out,
        slot_previous=0,
        slot_applied=1,
        slot_guard=2,
    )
    assert status == PREFIX_CACHE_ERR_POLICY
    assert out == [1, 2, 3]
    assert cache.entries[1].last_used_tick == 400


def test_strict_parity_rejects_bad_vectors():
    cache = _cache_with_tick(3)
    out = [0, 0, 0]

    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            None,
            1,
            2,
            PREFIX_CACHE_PROFILE_SECURE_LOCAL,
            out,
            0,
            1,
            2,
        )
        == PREFIX_CACHE_ERR_NULL_PTR
    )
    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            1,
            2,
            PREFIX_CACHE_PROFILE_SECURE_LOCAL,
            None,
            0,
            1,
            2,
        )
        == PREFIX_CACHE_ERR_NULL_PTR
    )
    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            1,
            2,
            PREFIX_CACHE_PROFILE_SECURE_LOCAL,
            out,
            0,
            0,
            2,
        )
        == PREFIX_CACHE_ERR_BAD_PARAM
    )
    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            -1,
            2,
            PREFIX_CACHE_PROFILE_SECURE_LOCAL,
            out,
            0,
            1,
            2,
        )
        == PREFIX_CACHE_ERR_BAD_PARAM
    )
    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            1,
            -1,
            PREFIX_CACHE_PROFILE_SECURE_LOCAL,
            out,
            0,
            1,
            2,
        )
        == PREFIX_CACHE_ERR_BAD_PARAM
    )
    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            cache,
            1,
            2,
            99,
            out,
            0,
            1,
            2,
        )
        == PREFIX_CACHE_ERR_BAD_PARAM
    )


if __name__ == "__main__":
    test_strict_parity_secure_local_monotonic_roundtrip()
    test_strict_parity_dev_local_allows_rollback()
    test_strict_parity_secure_local_policy_failure_keeps_outputs_unchanged()
    test_strict_parity_rejects_bad_vectors()
    print("ok")

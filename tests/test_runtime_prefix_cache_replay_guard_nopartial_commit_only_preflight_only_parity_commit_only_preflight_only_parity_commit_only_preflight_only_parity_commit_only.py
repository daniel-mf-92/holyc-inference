#!/usr/bin/env python3
"""Host-side harness for IQ-1358 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly."""

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
from test_runtime_prefix_cache_replay_guard_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)
from test_runtime_prefix_cache_replay_guard_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
)


def replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
    snapshot_entries_obj = cache.entries
    snapshot_entry_valid = cache.entries[entry_index].valid
    snapshot_previous_tick = cache.entries[entry_index].last_used_tick

    preflight_previous_seed = 0x13579BDF
    preflight_applied_seed = 0x2468ACE0
    preflight_guard_seed = 0x10203040
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        cache,
        entry_index,
        access_tick,
        profile_id,
        [preflight_previous_seed, preflight_applied_seed, preflight_guard_seed],
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
    if cache.entries is not snapshot_entries_obj:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].valid != snapshot_entry_valid:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if (
        entry_index != snapshot_entry_index
        or access_tick != snapshot_access_tick
        or profile_id != snapshot_profile_id
    ):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM

    cache.entries[entry_index].last_used_tick = snapshot_previous_tick

    parity_out = [0, 0, 0]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        cache,
        entry_index,
        access_tick,
        profile_id,
        parity_out,
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
    if cache.entries is not snapshot_entries_obj:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].valid != snapshot_entry_valid:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if (
        entry_index != snapshot_entry_index
        or access_tick != snapshot_access_tick
        or profile_id != snapshot_profile_id
    ):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_out[0] < 0 or parity_out[1] < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_out[2] not in (0, 1):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_out[0] != snapshot_previous_tick or parity_out[1] != access_tick or parity_out[2] != 1:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != parity_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM

    outputs[slot_previous] = parity_out[0]
    outputs[slot_applied] = parity_out[1]
    outputs[slot_guard] = parity_out[2]
    return PREFIX_CACHE_OK


def test_commit_only_diagnostics_secure_local_monotonic_roundtrip():
    cache = _cache_with_tick(700)
    out = [-1, -1, -1]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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


def test_commit_only_diagnostics_dev_local_allows_rollback():
    cache = _cache_with_tick(910)
    out = [99, 98, 97]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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


def test_commit_only_diagnostics_secure_local_policy_failure_keeps_output_slots_unchanged():
    cache = _cache_with_tick(400)
    out = [1, 2, 3]
    status = replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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


def test_commit_only_diagnostics_rejects_bad_vectors():
    cache = _cache_with_tick(3)
    out = [0, 0, 0]

    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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


def test_holyc_function_body_and_contract_markers_present():
    source = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    sig = "I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status_preflight = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(cache," in body
    assert "if (preflight_previous_tick_seed != 0x13579BDF ||" in body
    assert "cache->entries[entry_index].last_used_tick = snapshot_previous_tick;" in body
    assert "status_parity = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(cache," in body
    assert "if (parity_previous_tick != snapshot_previous_tick ||" in body
    assert "*out_previous_tick = parity_previous_tick;" in body
    assert "*out_applied_tick = parity_applied_tick;" in body
    assert "*out_guard_passed = parity_guard_passed;" in body


if __name__ == "__main__":
    test_commit_only_diagnostics_secure_local_monotonic_roundtrip()
    test_commit_only_diagnostics_dev_local_allows_rollback()
    test_commit_only_diagnostics_secure_local_policy_failure_keeps_output_slots_unchanged()
    test_commit_only_diagnostics_rejects_bad_vectors()
    test_holyc_function_body_and_contract_markers_present()
    print("ok")

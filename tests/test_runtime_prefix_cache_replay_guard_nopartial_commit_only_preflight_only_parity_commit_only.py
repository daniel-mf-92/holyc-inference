#!/usr/bin/env python3
"""Host-side harness for IQ-1335 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly."""

from dataclasses import dataclass
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4
PREFIX_CACHE_ERR_POLICY = 5

PREFIX_CACHE_PROFILE_SECURE_LOCAL = 1
PREFIX_CACHE_PROFILE_DEV_LOCAL = 2

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1


@dataclass
class PrefixCacheEntry:
    valid: int = PREFIX_CACHE_FRESH_EMPTY
    last_used_tick: int = 0


@dataclass
class PrefixCache:
    entries: list[PrefixCacheEntry] | None
    capacity: int
    count: int


def replay_guard_checked(cache: PrefixCache | None, entry_index: int, access_tick: int, profile_id: int):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if entry_index < 0 or entry_index >= cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    entry = cache.entries[entry_index]
    if entry.valid != PREFIX_CACHE_FRESH_VALID:
        return PREFIX_CACHE_ERR_NOT_FOUND, None, None, None

    if access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if profile_id not in (PREFIX_CACHE_PROFILE_SECURE_LOCAL, PREFIX_CACHE_PROFILE_DEV_LOCAL):
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    previous_tick = entry.last_used_tick
    if previous_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    if profile_id == PREFIX_CACHE_PROFILE_SECURE_LOCAL and access_tick < previous_tick:
        return PREFIX_CACHE_ERR_POLICY, None, None, None

    entry.last_used_tick = access_tick
    return PREFIX_CACHE_OK, previous_tick, access_tick, 1


def replay_guard_commit_only(
    cache: PrefixCache | None,
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

    snapshot_entry_index = entry_index
    snapshot_access_tick = access_tick
    snapshot_profile_id = profile_id
    snapshot_capacity = cache.capacity
    snapshot_count = cache.count

    status, staged_previous, staged_applied, staged_guard = replay_guard_checked(
        cache, entry_index, access_tick, profile_id
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
    if entry_index < 0 or entry_index >= cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if staged_previous is None or staged_applied is None or staged_guard is None:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if staged_previous < 0 or staged_applied < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if staged_applied != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != staged_applied:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if staged_guard not in (0, 1):
        return PREFIX_CACHE_ERR_BAD_PARAM

    outputs[slot_previous] = staged_previous
    outputs[slot_applied] = staged_applied
    outputs[slot_guard] = staged_guard
    return PREFIX_CACHE_OK


def replay_guard_commit_only_preflight_only(
    cache: PrefixCache | None,
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
    if cache.entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID:
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

    preflight_out = [0, 0, 0]
    status = replay_guard_commit_only(
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
    if preflight_out[1] != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != preflight_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM

    cache.entries[entry_index].last_used_tick = snapshot_previous_tick
    status, canonical_previous, canonical_applied, canonical_guard = replay_guard_checked(
        cache, entry_index, access_tick, profile_id
    )
    if status != PREFIX_CACHE_OK:
        return status

    if [canonical_previous, canonical_applied, canonical_guard] != preflight_out:
        return PREFIX_CACHE_ERR_BAD_PARAM

    outputs[slot_previous] = preflight_out[0]
    outputs[slot_applied] = preflight_out[1]
    outputs[slot_guard] = preflight_out[2]
    return PREFIX_CACHE_OK


def replay_guard_commit_only_preflight_only_parity(
    cache: PrefixCache | None,
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
    if cache.entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID:
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

    preflight_out = [0, 0, 0]
    status = replay_guard_commit_only_preflight_only(
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
    if preflight_out[1] != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != preflight_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM

    cache.entries[entry_index].last_used_tick = snapshot_previous_tick

    commit_out = [0, 0, 0]
    status = replay_guard_commit_only(
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
    if commit_out[1] != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != commit_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if preflight_out != commit_out:
        return PREFIX_CACHE_ERR_BAD_PARAM

    outputs[slot_previous] = preflight_out[0]
    outputs[slot_applied] = preflight_out[1]
    outputs[slot_guard] = preflight_out[2]
    return PREFIX_CACHE_OK


def replay_guard_commit_only_preflight_only_parity_commit_only(
    cache: PrefixCache | None,
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
    if cache.entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID:
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

    parity_out = [0, 0, 0]
    status = replay_guard_commit_only_preflight_only_parity(
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
    if (
        entry_index != snapshot_entry_index
        or access_tick != snapshot_access_tick
        or profile_id != snapshot_profile_id
    ):
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_out[1] != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != parity_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM

    cache.entries[entry_index].last_used_tick = snapshot_previous_tick

    commit_out = [0, 0, 0]
    status = replay_guard_commit_only_preflight_only(
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
    if commit_out[1] != access_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.entries[entry_index].last_used_tick != commit_out[1]:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if parity_out != commit_out:
        return PREFIX_CACHE_ERR_BAD_PARAM

    outputs[slot_previous] = parity_out[0]
    outputs[slot_applied] = parity_out[1]
    outputs[slot_guard] = parity_out[2]
    return PREFIX_CACHE_OK


def _cache_with_tick(tick: int) -> PrefixCache:
    entries = [PrefixCacheEntry() for _ in range(3)]
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, last_used_tick=tick)
    return PrefixCache(entries=entries, capacity=3, count=1)


def test_parity_commit_only_secure_local_monotonic_roundtrip():
    cache = _cache_with_tick(700)
    out = [-1, -1, -1]
    status = replay_guard_commit_only_preflight_only_parity_commit_only(
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


def test_parity_commit_only_dev_local_allows_rollback():
    cache = _cache_with_tick(910)
    out = [99, 98, 97]
    status = replay_guard_commit_only_preflight_only_parity_commit_only(
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


def test_parity_commit_only_secure_local_policy_failure_keeps_output_slots_unchanged():
    cache = _cache_with_tick(400)
    out = [1, 2, 3]
    status = replay_guard_commit_only_preflight_only_parity_commit_only(
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


def test_parity_commit_only_rejects_bad_vectors():
    cache = _cache_with_tick(3)
    out = [0, 0, 0]

    assert (
        replay_guard_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only(
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
        replay_guard_commit_only_preflight_only_parity_commit_only(
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
    sig = "I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status_parity = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParity(cache," in body
    assert "cache->entries[entry_index].last_used_tick = snapshot_previous_tick;" in body
    assert "status_commit = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnly(cache," in body
    assert "if (parity_previous_tick != commit_previous_tick ||" in body
    assert "*out_previous_tick = parity_previous_tick;" in body
    assert "*out_guard_passed = parity_guard_passed;" in body


if __name__ == "__main__":
    test_parity_commit_only_secure_local_monotonic_roundtrip()
    test_parity_commit_only_dev_local_allows_rollback()
    test_parity_commit_only_secure_local_policy_failure_keeps_output_slots_unchanged()
    test_parity_commit_only_rejects_bad_vectors()
    test_holyc_function_body_and_contract_markers_present()
    print("ok")

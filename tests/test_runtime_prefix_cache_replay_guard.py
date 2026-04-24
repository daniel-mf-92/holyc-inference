#!/usr/bin/env python3
"""Host-side harness for IQ-1291 PrefixCacheReplayGuardChecked."""

from dataclasses import dataclass
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_FULL = 3
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


def replay_guard_checked(
    cache: PrefixCache | None,
    entry_index: int,
    access_tick: int,
    profile_id: int,
):
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


def _cache_with_tick(tick: int) -> PrefixCache:
    entries = [PrefixCacheEntry() for _ in range(2)]
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, last_used_tick=tick)
    return PrefixCache(entries=entries, capacity=2, count=1)


def test_secure_local_accepts_monotonic_tick_and_writes_once():
    cache = _cache_with_tick(120)
    status, previous_tick, applied_tick, guard_passed = replay_guard_checked(
        cache,
        entry_index=1,
        access_tick=120,
        profile_id=PREFIX_CACHE_PROFILE_SECURE_LOCAL,
    )
    assert status == PREFIX_CACHE_OK
    assert previous_tick == 120
    assert applied_tick == 120
    assert guard_passed == 1
    assert cache.entries[1].last_used_tick == 120

    status, previous_tick, applied_tick, guard_passed = replay_guard_checked(
        cache,
        entry_index=1,
        access_tick=121,
        profile_id=PREFIX_CACHE_PROFILE_SECURE_LOCAL,
    )
    assert status == PREFIX_CACHE_OK
    assert previous_tick == 120
    assert applied_tick == 121
    assert guard_passed == 1
    assert cache.entries[1].last_used_tick == 121


def test_secure_local_blocks_rollback_without_mutating_tick():
    cache = _cache_with_tick(200)
    status, previous_tick, applied_tick, guard_passed = replay_guard_checked(
        cache,
        entry_index=1,
        access_tick=199,
        profile_id=PREFIX_CACHE_PROFILE_SECURE_LOCAL,
    )
    assert status == PREFIX_CACHE_ERR_POLICY
    assert previous_tick is None
    assert applied_tick is None
    assert guard_passed is None
    assert cache.entries[1].last_used_tick == 200


def test_dev_local_allows_tick_rollback_for_experimentation():
    cache = _cache_with_tick(300)
    status, previous_tick, applied_tick, guard_passed = replay_guard_checked(
        cache,
        entry_index=1,
        access_tick=120,
        profile_id=PREFIX_CACHE_PROFILE_DEV_LOCAL,
    )
    assert status == PREFIX_CACHE_OK
    assert previous_tick == 300
    assert applied_tick == 120
    assert guard_passed == 1
    assert cache.entries[1].last_used_tick == 120


def test_invalid_and_not_found_vectors():
    cache = _cache_with_tick(10)

    status, *_ = replay_guard_checked(None, 0, 0, PREFIX_CACHE_PROFILE_SECURE_LOCAL)
    assert status == PREFIX_CACHE_ERR_NULL_PTR

    bad_capacity_cache = PrefixCache(entries=[], capacity=0, count=0)
    status, *_ = replay_guard_checked(bad_capacity_cache, 0, 0, PREFIX_CACHE_PROFILE_SECURE_LOCAL)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    count_drift_cache = PrefixCache(entries=[PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID)], capacity=1, count=2)
    status, *_ = replay_guard_checked(count_drift_cache, 0, 0, PREFIX_CACHE_PROFILE_SECURE_LOCAL)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    status, *_ = replay_guard_checked(cache, -1, 1, PREFIX_CACHE_PROFILE_SECURE_LOCAL)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    status, *_ = replay_guard_checked(cache, 0, 1, PREFIX_CACHE_PROFILE_SECURE_LOCAL)
    assert status == PREFIX_CACHE_ERR_NOT_FOUND

    status, *_ = replay_guard_checked(cache, 1, -1, PREFIX_CACHE_PROFILE_SECURE_LOCAL)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    status, *_ = replay_guard_checked(cache, 1, 1, 999)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM


def test_holyc_function_body_and_contract_markers_present():
    source = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    sig = "I32 PrefixCacheReplayGuardChecked(PrefixCache *cache,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "profile_id == PREFIX_CACHE_PROFILE_SECURE_LOCAL" in body
    assert "access_tick < previous_tick" in body
    assert "return PREFIX_CACHE_ERR_POLICY;" in body
    assert "cache->entries[entry_index].last_used_tick = staged_applied_tick;" in body
    assert "*out_previous_tick = previous_tick;" in body
    assert "*out_guard_passed = staged_guard_passed;" in body


if __name__ == "__main__":
    test_secure_local_accepts_monotonic_tick_and_writes_once()
    test_secure_local_blocks_rollback_without_mutating_tick()
    test_dev_local_allows_tick_rollback_for_experimentation()
    test_invalid_and_not_found_vectors()
    test_holyc_function_body_and_contract_markers_present()
    print("ok")

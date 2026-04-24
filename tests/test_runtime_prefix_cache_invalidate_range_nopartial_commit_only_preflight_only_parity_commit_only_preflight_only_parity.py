#!/usr/bin/env python3
"""Host-side harness for IQ-1341 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1


@dataclass
class Entry:
    valid: int = PREFIX_CACHE_FRESH_EMPTY
    prefix_hash: int = 0
    prefix_tokens: int = 0
    kv_start_token: int = 0
    kv_token_count: int = 0
    last_used_tick: int = 0


@dataclass
class Cache:
    entries: list[Entry] | None
    capacity: int
    count: int


def _clone_entries(entries: list[Entry]) -> list[Entry]:
    return [Entry(**vars(entry)) for entry in entries]


def invalidate_range_checked(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    invalidate_end_exclusive = invalidate_start_token + invalidate_token_count
    if invalidate_end_exclusive < invalidate_start_token:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    live_count = 0
    removed_count = 0
    for entry in cache.entries:
        if entry.valid != PREFIX_CACHE_FRESH_VALID:
            continue
        if entry.kv_start_token < 0 or entry.kv_token_count < 0:
            return PREFIX_CACHE_ERR_BAD_PARAM, None

        entry_end_exclusive = entry.kv_start_token + entry.kv_token_count
        if entry_end_exclusive < entry.kv_start_token:
            return PREFIX_CACHE_ERR_BAD_PARAM, None

        overlap = (
            entry.kv_token_count > 0
            and invalidate_token_count > 0
            and invalidate_start_token < entry_end_exclusive
            and entry.kv_start_token < invalidate_end_exclusive
        )
        if overlap:
            entry.valid = PREFIX_CACHE_FRESH_EMPTY
            entry.prefix_hash = 0
            entry.prefix_tokens = 0
            entry.kv_start_token = 0
            entry.kv_token_count = 0
            entry.last_used_tick = 0
            removed_count += 1
        else:
            live_count += 1

    cache.count = live_count
    return PREFIX_CACHE_OK, removed_count


def invalidate_range_commit_only(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    invalidate_end_exclusive = invalidate_start_token + invalidate_token_count
    if invalidate_end_exclusive < invalidate_start_token:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    staged_live = 0
    staged_removed = 0
    for entry in cache.entries:
        if entry.valid != PREFIX_CACHE_FRESH_VALID:
            continue
        if entry.kv_start_token < 0 or entry.kv_token_count < 0:
            return PREFIX_CACHE_ERR_BAD_PARAM, None

        entry_end_exclusive = entry.kv_start_token + entry.kv_token_count
        if entry_end_exclusive < entry.kv_start_token:
            return PREFIX_CACHE_ERR_BAD_PARAM, None

        overlap = (
            entry.kv_token_count > 0
            and invalidate_token_count > 0
            and invalidate_start_token < entry_end_exclusive
            and entry.kv_start_token < invalidate_end_exclusive
        )
        if overlap:
            staged_removed += 1
        else:
            staged_live += 1

    snapshot_count = cache.count
    status, canonical_removed = invalidate_range_checked(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.count != staged_live:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if canonical_removed != staged_removed:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count > snapshot_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, staged_removed


def invalidate_range_commit_only_preflight_only(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_entries = _clone_entries(cache.entries)

    status, preflight_removed = invalidate_range_commit_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    preflight_remaining = cache.count
    if preflight_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed < 0 or preflight_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed + preflight_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, canonical_removed = invalidate_range_checked(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    canonical_remaining = cache.count
    if canonical_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if canonical_removed < 0 or canonical_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if canonical_removed + canonical_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    if preflight_removed != canonical_removed or preflight_remaining != canonical_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, preflight_removed


def invalidate_range_commit_only_preflight_only_parity(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_entries = _clone_entries(cache.entries)

    status, preflight_removed = invalidate_range_commit_only_preflight_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    preflight_remaining = cache.count
    if preflight_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed < 0 or preflight_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed > snapshot_entry_count or preflight_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed + preflight_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, commit_removed = invalidate_range_commit_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    commit_remaining = cache.count
    if commit_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed < 0 or commit_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed > snapshot_entry_count or commit_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed + commit_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    if preflight_removed != commit_removed or preflight_remaining != commit_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, preflight_removed


def invalidate_range_commit_only_preflight_only_parity_commit_only(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_entries = _clone_entries(cache.entries)

    status, parity_removed = invalidate_range_commit_only_preflight_only_parity(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    parity_remaining = cache.count
    if parity_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed < 0 or parity_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed > snapshot_entry_count or parity_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed + parity_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, commit_removed = invalidate_range_commit_only_preflight_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    commit_remaining = cache.count
    if commit_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed < 0 or commit_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed > snapshot_entry_count or commit_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed + commit_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    if parity_removed != commit_removed or parity_remaining != commit_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, parity_removed


def invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_entries = _clone_entries(cache.entries)

    status, commit_removed = invalidate_range_commit_only_preflight_only_parity_commit_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    commit_remaining = cache.count
    if commit_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed < 0 or commit_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed > snapshot_entry_count or commit_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed + commit_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, parity_removed = invalidate_range_commit_only_preflight_only_parity(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    parity_remaining = cache.count
    if parity_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed < 0 or parity_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed > snapshot_entry_count or parity_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if parity_removed + parity_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    if commit_removed != parity_removed or commit_remaining != parity_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, commit_removed


def invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    cache: Cache | None,
    invalidate_start_token: int,
    invalidate_token_count: int,
):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token < 0 or invalidate_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    snapshot_capacity = cache.capacity
    snapshot_count = cache.count
    snapshot_start = invalidate_start_token
    snapshot_tokens = invalidate_token_count
    snapshot_entry_count = cache.count
    snapshot_entries = _clone_entries(cache.entries)

    status, preflight_removed = invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    preflight_remaining = cache.count
    if preflight_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed < 0 or preflight_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed > snapshot_entry_count or preflight_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if preflight_removed + preflight_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    cache.entries = _clone_entries(snapshot_entries)
    cache.count = snapshot_count

    status, commit_removed = invalidate_range_commit_only_preflight_only_parity_commit_only(
        cache,
        invalidate_start_token,
        invalidate_token_count,
    )
    if status != PREFIX_CACHE_OK:
        return status, None

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if invalidate_start_token != snapshot_start or invalidate_token_count != snapshot_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    commit_remaining = cache.count
    if commit_removed is None:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed < 0 or commit_remaining < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed > snapshot_entry_count or commit_remaining > snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if commit_removed + commit_remaining != snapshot_entry_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    if preflight_removed != commit_removed or preflight_remaining != commit_remaining:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    return PREFIX_CACHE_OK, preflight_removed


def test_iq1341_preflight_only_parity_matches_commit_wrapper_tuple():
    entries = [Entry() for _ in range(6)]
    entries[0] = Entry(valid=1, prefix_hash=11, prefix_tokens=8, kv_start_token=0, kv_token_count=16)
    entries[1] = Entry(valid=1, prefix_hash=12, prefix_tokens=12, kv_start_token=16, kv_token_count=16)
    entries[2] = Entry(valid=1, prefix_hash=13, prefix_tokens=20, kv_start_token=40, kv_token_count=10)
    entries[4] = Entry(valid=1, prefix_hash=14, prefix_tokens=28, kv_start_token=60, kv_token_count=8)
    cache = Cache(entries=entries, capacity=6, count=4)

    status, removed = invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity(cache, 12, 30)
    assert status == PREFIX_CACHE_OK
    assert removed == 3
    assert cache.count == 1
    assert entries[4].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[0].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[1].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[2].valid == PREFIX_CACHE_FRESH_EMPTY


def test_iq1341_preflight_only_parity_zero_length_invalidate_is_noop():
    entries = [Entry() for _ in range(3)]
    entries[0] = Entry(valid=1, prefix_hash=31, prefix_tokens=4, kv_start_token=10, kv_token_count=5)
    entries[1] = Entry(valid=1, prefix_hash=32, prefix_tokens=5, kv_start_token=20, kv_token_count=5)
    cache = Cache(entries=entries, capacity=3, count=2)

    status, removed = invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity(cache, 12, 0)
    assert status == PREFIX_CACHE_OK
    assert removed == 0
    assert cache.count == 2
    assert entries[0].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[1].valid == PREFIX_CACHE_FRESH_VALID


def test_iq1341_preflight_only_parity_guard_keeps_state_unchanged_on_error():
    entries = [Entry() for _ in range(2)]
    entries[0] = Entry(valid=1, prefix_hash=21, prefix_tokens=4, kv_start_token=0, kv_token_count=8)
    entries[1] = Entry(valid=1, prefix_hash=22, prefix_tokens=5, kv_start_token=-3, kv_token_count=5)
    cache = Cache(entries=entries, capacity=2, count=2)
    before = _clone_entries(entries)

    status, removed = invalidate_range_commit_only_preflight_only_parity_commit_only_preflight_only_parity(cache, 0, 10)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM
    assert removed is None
    assert cache.count == 2
    for idx, entry in enumerate(entries):
        assert vars(entry) == vars(before[idx])


def test_holyc_function_body_and_contract_markers_present():
    source = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    sig = (
        "I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(PrefixCache *cache,"
    )
    assert sig in source
    body = source.split(sig, 1)[1]
    assert (
        "status_preflight = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(cache,"
        in body
    )
    assert "cache->entries[idx] = snapshot_entries[idx];" in body
    assert (
        "status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(cache," in body
    )
    assert "if (preflight_removed_count != commit_removed_count ||" in body
    assert "*out_removed_count = preflight_removed_count;" in body


if __name__ == "__main__":
    test_iq1341_preflight_only_parity_matches_commit_wrapper_tuple()
    test_iq1341_preflight_only_parity_zero_length_invalidate_is_noop()
    test_iq1341_preflight_only_parity_guard_keeps_state_unchanged_on_error()
    test_holyc_function_body_and_contract_markers_present()
    print("ok")

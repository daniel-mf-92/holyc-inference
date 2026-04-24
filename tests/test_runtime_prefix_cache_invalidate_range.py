#!/usr/bin/env python3
"""Host-side harness for IQ-1289 PrefixCacheInvalidateRangeChecked."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "src" / "runtime" / "prefix_cache.HC"


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


def test_iq1289_invalidates_all_overlapping_windows():
    entries = [Entry() for _ in range(5)]
    entries[0] = Entry(valid=1, prefix_hash=11, prefix_tokens=8, kv_start_token=0, kv_token_count=16)
    entries[1] = Entry(valid=1, prefix_hash=12, prefix_tokens=12, kv_start_token=16, kv_token_count=16)
    entries[2] = Entry(valid=1, prefix_hash=13, prefix_tokens=20, kv_start_token=40, kv_token_count=10)
    entries[3] = Entry(valid=1, prefix_hash=14, prefix_tokens=28, kv_start_token=60, kv_token_count=8)
    cache = Cache(entries=entries, capacity=len(entries), count=4)

    status, removed = invalidate_range_checked(cache, invalidate_start_token=12, invalidate_token_count=30)
    assert status == PREFIX_CACHE_OK
    assert removed == 3
    assert cache.count == 1
    assert entries[3].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[0].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[1].valid == PREFIX_CACHE_FRESH_EMPTY
    assert entries[2].valid == PREFIX_CACHE_FRESH_EMPTY


def test_iq1289_zero_length_invalidation_is_noop():
    entries = [Entry() for _ in range(3)]
    entries[0] = Entry(valid=1, prefix_hash=21, prefix_tokens=4, kv_start_token=10, kv_token_count=5)
    entries[1] = Entry(valid=1, prefix_hash=22, prefix_tokens=5, kv_start_token=20, kv_token_count=5)
    cache = Cache(entries=entries, capacity=len(entries), count=2)

    status, removed = invalidate_range_checked(cache, invalidate_start_token=12, invalidate_token_count=0)
    assert status == PREFIX_CACHE_OK
    assert removed == 0
    assert cache.count == 2
    assert entries[0].valid == PREFIX_CACHE_FRESH_VALID
    assert entries[1].valid == PREFIX_CACHE_FRESH_VALID


def test_iq1289_guard_paths():
    entries = [Entry(valid=1, kv_start_token=0, kv_token_count=8)]
    cache = Cache(entries=entries, capacity=1, count=1)

    assert invalidate_range_checked(None, 0, 1)[0] == PREFIX_CACHE_ERR_NULL_PTR
    assert invalidate_range_checked(Cache(entries=None, capacity=1, count=0), 0, 1)[0] == PREFIX_CACHE_ERR_NULL_PTR
    assert invalidate_range_checked(Cache(entries=entries, capacity=0, count=1), 0, 1)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert invalidate_range_checked(cache, -1, 1)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert invalidate_range_checked(cache, 0, -1)[0] == PREFIX_CACHE_ERR_BAD_PARAM


def test_iq1289_source_contains_signature_and_overlap_clause():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    signature = (
        "I32 PrefixCacheInvalidateRangeChecked(PrefixCache *cache,\n"
        "                                      I64 invalidate_start_token,\n"
        "                                      I64 invalidate_token_count,\n"
        "                                      I64 *out_removed_count)\n"
    )
    assert signature in source
    assert "invalidate_start_token < entry_end_exclusive" in source
    assert "entry_start_token < invalidate_end_exclusive" in source


if __name__ == "__main__":
    test_iq1289_invalidates_all_overlapping_windows()
    test_iq1289_zero_length_invalidation_is_noop()
    test_iq1289_guard_paths()
    test_iq1289_source_contains_signature_and_overlap_clause()
    print("ok")

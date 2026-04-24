#!/usr/bin/env python3
"""Host-side harness for PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnly (IQ-1292)."""

from __future__ import annotations

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1


class Entry:
    __slots__ = (
        "valid",
        "prefix_hash",
        "prefix_tokens",
        "kv_start_token",
        "kv_token_count",
        "last_used_tick",
    )

    def __init__(self):
        self.valid = PREFIX_CACHE_FRESH_EMPTY
        self.prefix_hash = 0
        self.prefix_tokens = 0
        self.kv_start_token = 0
        self.kv_token_count = 0
        self.last_used_tick = 0


class Cache:
    def __init__(self, capacity: int):
        self.entries = [Entry() for _ in range(max(capacity, 0))]
        self.capacity = capacity
        self.count = 0


def validate_tuple(prefix_hash: int, prefix_tokens: int, kv_start: int, kv_count: int) -> int:
    if prefix_hash < 0 or prefix_tokens < 0 or kv_start < 0 or kv_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    return PREFIX_CACHE_OK


def find_index(cache: Cache, prefix_hash: int, prefix_tokens: int):
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    for idx, entry in enumerate(cache.entries):
        if (
            entry.valid == PREFIX_CACHE_FRESH_VALID
            and entry.prefix_hash == prefix_hash
            and entry.prefix_tokens == prefix_tokens
        ):
            return PREFIX_CACHE_OK, idx
    return PREFIX_CACHE_ERR_NOT_FOUND, None


def select_lru_victim(cache: Cache):
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    for idx, entry in enumerate(cache.entries):
        if entry.valid != PREFIX_CACHE_FRESH_VALID:
            return PREFIX_CACHE_OK, idx

    oldest_idx = 0
    oldest_tick = cache.entries[0].last_used_tick
    for idx in range(1, cache.capacity):
        tick = cache.entries[idx].last_used_tick
        if tick < oldest_tick:
            oldest_tick = tick
            oldest_idx = idx
    return PREFIX_CACHE_OK, oldest_idx


def insert_or_update_checked(cache: Cache, prefix_hash: int, prefix_tokens: int, kv_start: int, kv_count: int, access_tick: int):
    if cache.capacity <= 0 or access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count) != PREFIX_CACHE_OK:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    status, found = find_index(cache, prefix_hash, prefix_tokens)
    if status == PREFIX_CACHE_OK:
        entry = cache.entries[found]
        entry.kv_start_token = kv_start
        entry.kv_token_count = kv_count
        entry.last_used_tick = access_tick
        return PREFIX_CACHE_OK, found, 0
    if status != PREFIX_CACHE_ERR_NOT_FOUND:
        return status, None, None

    status, victim = select_lru_victim(cache)
    if status != PREFIX_CACHE_OK:
        return status, None, None

    if cache.entries[victim].valid != PREFIX_CACHE_FRESH_VALID:
        cache.count += 1

    entry = cache.entries[victim]
    entry.valid = PREFIX_CACHE_FRESH_VALID
    entry.prefix_hash = prefix_hash
    entry.prefix_tokens = prefix_tokens
    entry.kv_start_token = kv_start
    entry.kv_token_count = kv_count
    entry.last_used_tick = access_tick
    return PREFIX_CACHE_OK, victim, 1


def commit_only_insert_or_update(cache: Cache, prefix_hash: int, prefix_tokens: int, kv_start: int, kv_count: int, access_tick: int, out_pair):
    if cache is None or out_pair is None:
        return PREFIX_CACHE_ERR_NULL_PTR
    if cache.capacity <= 0 or access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count) != PREFIX_CACHE_OK:
        return PREFIX_CACHE_ERR_BAD_PARAM

    snapshot_prefix_hash = prefix_hash
    snapshot_prefix_tokens = prefix_tokens
    snapshot_kv_start = kv_start
    snapshot_kv_count = kv_count
    snapshot_count = cache.count
    snapshot_capacity = cache.capacity
    snapshot_tick = access_tick

    status, entry_index, inserted_new = insert_or_update_checked(
        cache,
        prefix_hash,
        prefix_tokens,
        kv_start,
        kv_count,
        access_tick,
    )
    if status != PREFIX_CACHE_OK:
        return status

    if cache.capacity != snapshot_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if (
        prefix_hash != snapshot_prefix_hash
        or prefix_tokens != snapshot_prefix_tokens
        or kv_start != snapshot_kv_start
        or kv_count != snapshot_kv_count
    ):
        return PREFIX_CACHE_ERR_BAD_PARAM

    if inserted_new == 0:
        if cache.count != snapshot_count:
            return PREFIX_CACHE_ERR_BAD_PARAM
    else:
        if cache.count not in (snapshot_count, snapshot_count + 1):
            return PREFIX_CACHE_ERR_BAD_PARAM

    if access_tick != snapshot_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if entry_index < 0 or entry_index >= cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if inserted_new not in (0, 1):
        return PREFIX_CACHE_ERR_BAD_PARAM

    out_pair[0] = entry_index
    out_pair[1] = inserted_new
    return PREFIX_CACHE_OK


def test_commit_only_insert_then_update_roundtrip():
    cache = Cache(2)
    out = [99, 99]

    assert commit_only_insert_or_update(cache, 101, 4, 0, 4, 10, out) == PREFIX_CACHE_OK
    assert out == [0, 1]
    assert cache.count == 1

    assert commit_only_insert_or_update(cache, 101, 4, 4, 4, 11, out) == PREFIX_CACHE_OK
    assert out == [0, 0]
    assert cache.count == 1
    assert cache.entries[0].kv_start_token == 4
    assert cache.entries[0].kv_token_count == 4
    assert cache.entries[0].last_used_tick == 11


def test_commit_only_lru_insert_allows_replacement_without_count_drift():
    cache = Cache(2)
    out = [77, 77]

    assert commit_only_insert_or_update(cache, 10, 2, 0, 2, 5, out) == PREFIX_CACHE_OK
    assert commit_only_insert_or_update(cache, 20, 3, 2, 3, 6, out) == PREFIX_CACHE_OK
    assert cache.count == 2

    # Make slot 0 oldest, then force replacement.
    cache.entries[0].last_used_tick = 1
    cache.entries[1].last_used_tick = 9

    assert commit_only_insert_or_update(cache, 30, 4, 5, 4, 10, out) == PREFIX_CACHE_OK
    assert out == [0, 1]
    assert cache.count == 2
    assert cache.entries[0].prefix_hash == 30


def test_commit_only_rejects_bad_inputs():
    out = [0, 0]

    assert commit_only_insert_or_update(None, 1, 1, 1, 1, 1, out) == PREFIX_CACHE_ERR_NULL_PTR
    assert commit_only_insert_or_update(Cache(2), 1, 1, 1, 1, 1, None) == PREFIX_CACHE_ERR_NULL_PTR

    bad = Cache(0)
    assert commit_only_insert_or_update(bad, 1, 1, 1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM

    cache = Cache(2)
    assert commit_only_insert_or_update(cache, -1, 1, 1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert commit_only_insert_or_update(cache, 1, -1, 1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert commit_only_insert_or_update(cache, 1, 1, -1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert commit_only_insert_or_update(cache, 1, 1, 1, -1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert commit_only_insert_or_update(cache, 1, 1, 1, 1, -1, out) == PREFIX_CACHE_ERR_BAD_PARAM


def main():
    test_commit_only_insert_then_update_roundtrip()
    test_commit_only_lru_insert_allows_replacement_without_count_drift()
    test_commit_only_rejects_bad_inputs()
    print("ok")


if __name__ == "__main__":
    main()

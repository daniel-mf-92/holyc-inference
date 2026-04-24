#!/usr/bin/env python3
"""Parity/no-partial harness for IQ-1300 prefix-cache preflight-only wrapper."""

from __future__ import annotations

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_FULL = 3
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
        self.entries = [Entry() for _ in range(capacity)]
        self.capacity = capacity
        self.count = 0



def validate_tuple(prefix_hash, prefix_tokens, kv_start_token, kv_token_count):
    if prefix_hash < 0 or prefix_tokens < 0 or kv_start_token < 0 or kv_token_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    return PREFIX_CACHE_OK


def find_index(cache: Cache, prefix_hash: int, prefix_tokens: int):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
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
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None

    for idx, entry in enumerate(cache.entries):
        if entry.valid != PREFIX_CACHE_FRESH_VALID:
            return PREFIX_CACHE_OK, idx

    best_idx = 0
    best_tick = cache.entries[0].last_used_tick
    for idx in range(1, cache.capacity):
        tick = cache.entries[idx].last_used_tick
        if tick < best_tick:
            best_tick = tick
            best_idx = idx
    return PREFIX_CACHE_OK, best_idx


def canonical_insert_or_update(cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0 or access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count) != PREFIX_CACHE_OK:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    status, found = find_index(cache, prefix_hash, prefix_tokens)
    if status == PREFIX_CACHE_OK:
        e = cache.entries[found]
        e.kv_start_token = kv_start
        e.kv_token_count = kv_count
        e.last_used_tick = access_tick
        return PREFIX_CACHE_OK, found, 0
    if status != PREFIX_CACHE_ERR_NOT_FOUND:
        return status, None, None

    status, victim = select_lru_victim(cache)
    if status != PREFIX_CACHE_OK:
        return status, None, None

    if cache.entries[victim].valid != PREFIX_CACHE_FRESH_VALID:
        cache.count += 1

    e = cache.entries[victim]
    e.valid = PREFIX_CACHE_FRESH_VALID
    e.prefix_hash = prefix_hash
    e.prefix_tokens = prefix_tokens
    e.kv_start_token = kv_start
    e.kv_token_count = kv_count
    e.last_used_tick = access_tick
    return PREFIX_CACHE_OK, victim, 1


def commit_only_insert_or_update(cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick, out_pair):
    if cache is None or out_pair is None:
        return PREFIX_CACHE_ERR_NULL_PTR
    if cache.capacity <= 0 or access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM

    count_snapshot = cache.count
    capacity_snapshot = cache.capacity
    tick_snapshot = access_tick

    status, idx, ins = canonical_insert_or_update(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if status != PREFIX_CACHE_OK:
        return status

    if cache.capacity != capacity_snapshot:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if ins == 0:
        if cache.count != count_snapshot:
            return PREFIX_CACHE_ERR_BAD_PARAM
    else:
        if cache.count not in (count_snapshot, count_snapshot + 1):
            return PREFIX_CACHE_ERR_BAD_PARAM

    if access_tick != tick_snapshot:
        return PREFIX_CACHE_ERR_BAD_PARAM

    out_pair[0] = idx
    out_pair[1] = ins
    return PREFIX_CACHE_OK


def preflight_only_insert_or_update(cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick, out_pair):
    if cache is None or out_pair is None:
        return PREFIX_CACHE_ERR_NULL_PTR
    if cache.capacity <= 0 or access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count) != PREFIX_CACHE_OK:
        return PREFIX_CACHE_ERR_BAD_PARAM

    capacity_snapshot = cache.capacity
    tick_snapshot = access_tick

    preflight_pair = [0, 0]
    status = commit_only_insert_or_update(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick, preflight_pair
    )
    if status != PREFIX_CACHE_OK:
        return status

    status, canonical_idx, canonical_ins = canonical_insert_or_update(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if status != PREFIX_CACHE_OK:
        return status

    if preflight_pair[0] != canonical_idx or preflight_pair[1] != canonical_ins:
        return PREFIX_CACHE_ERR_BAD_PARAM

    if cache.capacity != capacity_snapshot:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM
    if access_tick != tick_snapshot:
        return PREFIX_CACHE_ERR_BAD_PARAM

    out_pair[0] = preflight_pair[0]
    out_pair[1] = preflight_pair[1]
    return PREFIX_CACHE_OK


def run_ok_case(cache, args):
    out = [9999, 9999]
    status = preflight_only_insert_or_update(cache, *args, out)
    assert status == PREFIX_CACHE_OK
    assert 0 <= out[0] < cache.capacity
    assert out[1] in (0, 1)


def main():
    cache = Cache(2)

    run_ok_case(cache, (123, 4, 0, 4, 10))
    run_ok_case(cache, (123, 4, 1, 4, 11))
    run_ok_case(cache, (999, 6, 4, 6, 20))
    run_ok_case(cache, (777, 8, 10, 8, 1))

    out = [777, 888]
    assert preflight_only_insert_or_update(None, 1, 1, 1, 1, 1, out) == PREFIX_CACHE_ERR_NULL_PTR
    assert preflight_only_insert_or_update(Cache(0), 1, 1, 1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert preflight_only_insert_or_update(Cache(2), -1, 1, 1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert preflight_only_insert_or_update(Cache(2), 1, -1, 1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert preflight_only_insert_or_update(Cache(2), 1, 1, -1, 1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert preflight_only_insert_or_update(Cache(2), 1, 1, 1, -1, 1, out) == PREFIX_CACHE_ERR_BAD_PARAM
    assert preflight_only_insert_or_update(Cache(2), 1, 1, 1, 1, -1, out) == PREFIX_CACHE_ERR_BAD_PARAM

    print("ok")


if __name__ == "__main__":
    main()

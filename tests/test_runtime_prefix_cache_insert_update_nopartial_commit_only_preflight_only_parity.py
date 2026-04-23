#!/usr/bin/env python3
"""Parity harness for IQ-1301 PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnlyPreflightOnlyParity.

Host-side deterministic model mirrors HolyC semantics with integer-only state.
"""

from dataclasses import dataclass

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4

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
    entries: list
    capacity: int
    count: int = 0


def make_cache(capacity: int) -> Cache:
    return Cache(entries=[Entry() for _ in range(capacity)], capacity=capacity, count=0)


def _validate_tuple(prefix_hash: int, prefix_tokens: int, kv_start: int, kv_count: int) -> int:
    if prefix_hash < 0 or prefix_tokens < 0 or kv_start < 0 or kv_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM
    return PREFIX_CACHE_OK


def _find_index(cache: Cache, prefix_hash: int, prefix_tokens: int):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    if prefix_hash < 0 or prefix_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    for i, e in enumerate(cache.entries):
        if e.valid == PREFIX_CACHE_FRESH_VALID and e.prefix_hash == prefix_hash and e.prefix_tokens == prefix_tokens:
            return PREFIX_CACHE_OK, i
    return PREFIX_CACHE_ERR_NOT_FOUND, None


def _select_victim_lru(cache: Cache):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    for i, e in enumerate(cache.entries):
        if e.valid != PREFIX_CACHE_FRESH_VALID:
            return PREFIX_CACHE_OK, i
    oldest_tick = cache.entries[0].last_used_tick
    victim = 0
    for i in range(1, cache.capacity):
        if cache.entries[i].last_used_tick < oldest_tick:
            oldest_tick = cache.entries[i].last_used_tick
            victim = i
    return PREFIX_CACHE_OK, victim


def insert_or_update_checked(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    st = _validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count)
    if st != PREFIX_CACHE_OK:
        return st, None, None

    st, found = _find_index(cache, prefix_hash, prefix_tokens)
    if st == PREFIX_CACHE_OK:
        e = cache.entries[found]
        e.kv_start_token = kv_start
        e.kv_token_count = kv_count
        e.last_used_tick = access_tick
        return PREFIX_CACHE_OK, found, 0
    if st != PREFIX_CACHE_ERR_NOT_FOUND:
        return st, None, None

    st, victim = _select_victim_lru(cache)
    if st != PREFIX_CACHE_OK:
        return st, None, None

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


def insert_or_update_commit_only(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if access_tick < 0 or prefix_hash < 0 or prefix_tokens < 0 or kv_start < 0 or kv_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snap_count = cache.count
    snap_capacity = cache.capacity
    snap_access = access_tick

    st, idx, inserted = insert_or_update_checked(cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick)
    if st != PREFIX_CACHE_OK:
        return st, None, None

    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if cache.capacity != snap_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if access_tick != snap_access:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if inserted == 0 and cache.count != snap_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if inserted == 1 and cache.count != snap_count and cache.count != snap_count + 1:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, idx, inserted


def insert_or_update_preflight_only(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if access_tick < 0 or prefix_hash < 0 or prefix_tokens < 0 or kv_start < 0 or kv_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snap_entries = [Entry(**vars(e)) for e in cache.entries]
    snap_count = cache.count

    st_pre, idx_pre, ins_pre = insert_or_update_commit_only(cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick)
    if st_pre != PREFIX_CACHE_OK:
        return st_pre, None, None

    snap_after_preflight_entries = [Entry(**vars(e)) for e in cache.entries]
    snap_after_preflight_count = cache.count

    st_can, idx_can, ins_can = insert_or_update_checked(cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick)
    if st_can != PREFIX_CACHE_OK:
        return st_can, None, None

    if idx_pre != idx_can or ins_pre != ins_can:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    if cache.count != snap_after_preflight_count or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    for now, before in zip(cache.entries, snap_after_preflight_entries):
        if vars(now) != vars(before):
            return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, idx_pre, ins_pre


def insert_or_update_preflight_only_parity(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if access_tick < 0 or prefix_hash < 0 or prefix_tokens < 0 or kv_start < 0 or kv_count < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snap_count = cache.count
    snap_capacity = cache.capacity
    snap_access = access_tick

    st_pre, idx_pre, ins_pre = insert_or_update_preflight_only(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if st_pre != PREFIX_CACHE_OK:
        return st_pre, None, None

    snap_after_parity_entries = [Entry(**vars(e)) for e in cache.entries]
    snap_after_parity_count = cache.count

    st_can, idx_can, ins_can = insert_or_update_commit_only(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if st_can != PREFIX_CACHE_OK:
        return st_can, None, None

    if idx_pre != idx_can or ins_pre != ins_can:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if cache.capacity != snap_capacity or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if access_tick != snap_access:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if cache.count != snap_after_parity_count or cache.count < snap_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    for now, before in zip(cache.entries, snap_after_parity_entries):
        if vars(now) != vars(before):
            return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, idx_pre, ins_pre


def test_insert_new_parity_ok():
    c = make_cache(2)
    st, idx, inserted = insert_or_update_preflight_only_parity(c, 101, 8, 0, 8, 1)
    assert st == PREFIX_CACHE_OK
    assert idx == 0
    assert inserted == 1


def test_update_existing_parity_ok():
    c = make_cache(2)
    st, idx, inserted = insert_or_update_preflight_only_parity(c, 101, 8, 0, 8, 1)
    assert st == PREFIX_CACHE_OK and inserted == 1
    st, idx2, inserted2 = insert_or_update_preflight_only_parity(c, 101, 8, 8, 8, 2)
    assert st == PREFIX_CACHE_OK
    assert idx2 == idx
    assert inserted2 == 0
    assert c.entries[idx].kv_start_token == 8


def test_lru_replacement_path_ok():
    c = make_cache(2)
    assert insert_or_update_preflight_only_parity(c, 100, 4, 0, 4, 10)[0] == PREFIX_CACHE_OK
    assert insert_or_update_preflight_only_parity(c, 101, 5, 4, 5, 20)[0] == PREFIX_CACHE_OK
    st, idx, inserted = insert_or_update_preflight_only_parity(c, 102, 6, 9, 6, 30)
    assert st == PREFIX_CACHE_OK
    assert inserted == 1
    assert idx == 0


def test_bad_param_rejected():
    c = make_cache(2)
    st, _, _ = insert_or_update_preflight_only_parity(c, -1, 8, 0, 8, 1)
    assert st == PREFIX_CACHE_ERR_BAD_PARAM
    st, _, _ = insert_or_update_preflight_only_parity(c, 1, 8, 0, 8, -1)
    assert st == PREFIX_CACHE_ERR_BAD_PARAM


if __name__ == "__main__":
    test_insert_new_parity_ok()
    test_update_existing_parity_ok()
    test_lru_replacement_path_ok()
    test_bad_param_rejected()
    print("ok")

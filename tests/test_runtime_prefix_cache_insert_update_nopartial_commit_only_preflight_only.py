#!/usr/bin/env python3
"""Host-side harness for IQ-1300.

Validates HolyC semantics for
PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnlyPreflightOnly:
- runs commit-only then canonical insert/update on same cache
- enforces strict tuple parity {slot_index,new_generation,new_token_count}
- diagnostics wrapper is zero-write for all *published outputs*
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
    generation: int = 0


@dataclass
class Cache:
    entries: list
    capacity: int
    count: int = 0
    global_generation: int = 0


def make_cache(capacity: int) -> Cache:
    return Cache(entries=[Entry() for _ in range(capacity)], capacity=capacity)


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

    oldest = cache.entries[0].last_used_tick
    victim = 0
    for i in range(1, cache.capacity):
        if cache.entries[i].last_used_tick < oldest:
            oldest = cache.entries[i].last_used_tick
            victim = i
    return PREFIX_CACHE_OK, victim


def _next_generation(cache: Cache):
    if cache.global_generation >= (1 << 62):
        return PREFIX_CACHE_ERR_BAD_PARAM, None
    cache.global_generation += 1
    return PREFIX_CACHE_OK, cache.global_generation


def insert_or_update_checked(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    st = _validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count)
    if st != PREFIX_CACHE_OK:
        return st, None, None, None

    st, idx = _find_index(cache, prefix_hash, prefix_tokens)
    if st == PREFIX_CACHE_OK:
        e = cache.entries[idx]
        st_g, g = _next_generation(cache)
        if st_g != PREFIX_CACHE_OK:
            return st_g, None, None, None

        e.kv_start_token = kv_start
        e.kv_token_count = kv_count
        e.last_used_tick = access_tick
        e.generation = g
        return PREFIX_CACHE_OK, idx, g, e.kv_token_count

    if st != PREFIX_CACHE_ERR_NOT_FOUND:
        return st, None, None, None

    st, victim = _select_victim_lru(cache)
    if st != PREFIX_CACHE_OK:
        return st, None, None, None

    if cache.entries[victim].valid != PREFIX_CACHE_FRESH_VALID:
        cache.count += 1

    st_g, g = _next_generation(cache)
    if st_g != PREFIX_CACHE_OK:
        return st_g, None, None, None

    e = cache.entries[victim]
    e.valid = PREFIX_CACHE_FRESH_VALID
    e.prefix_hash = prefix_hash
    e.prefix_tokens = prefix_tokens
    e.kv_start_token = kv_start
    e.kv_token_count = kv_count
    e.last_used_tick = access_tick
    e.generation = g

    return PREFIX_CACHE_OK, victim, g, e.kv_token_count


def insert_or_update_commit_only(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    st = _validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count)
    if st != PREFIX_CACHE_OK:
        return st, None, None, None

    snap_count = cache.count
    snap_capacity = cache.capacity
    snap_tick = access_tick

    st, slot, new_gen, new_count = insert_or_update_checked(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if st != PREFIX_CACHE_OK:
        return st, None, None, None

    if cache.capacity != snap_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if access_tick != snap_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if cache.count != snap_count and cache.count != snap_count + 1:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    e = cache.entries[slot]
    if e.valid != PREFIX_CACHE_FRESH_VALID:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if e.generation != new_gen:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if e.kv_token_count != new_count:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    return PREFIX_CACHE_OK, slot, new_gen, new_count


def insert_or_update_preflight_only(
    cache: Cache,
    prefix_hash: int,
    prefix_tokens: int,
    kv_start: int,
    kv_count: int,
    access_tick: int,
):
    if cache is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if access_tick < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    st = _validate_tuple(prefix_hash, prefix_tokens, kv_start, kv_count)
    if st != PREFIX_CACHE_OK:
        return st, None, None, None

    snap_capacity = cache.capacity
    snap_tick = access_tick

    st_pre, slot_pre, gen_pre, tok_pre = insert_or_update_commit_only(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if st_pre != PREFIX_CACHE_OK:
        return st_pre, None, None, None

    st_can, slot_can, gen_can, tok_can = insert_or_update_checked(
        cache, prefix_hash, prefix_tokens, kv_start, kv_count, access_tick
    )
    if st_can != PREFIX_CACHE_OK:
        return st_can, None, None, None

    if slot_pre != slot_can or gen_pre != gen_can or tok_pre != tok_can:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    if cache.capacity != snap_capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None
    if access_tick != snap_tick:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None, None

    return PREFIX_CACHE_OK, slot_pre, gen_pre, tok_pre


def test_new_insert_rejected_by_strict_parity():
    c = make_cache(2)
    st, slot, gen, tok = insert_or_update_preflight_only(c, 101, 8, 0, 8, 1)
    assert st == PREFIX_CACHE_ERR_BAD_PARAM
    assert slot is None and gen is None and tok is None


def test_existing_update_passes_strict_parity():
    c = make_cache(2)
    st_seed, idx, _, _ = insert_or_update_commit_only(c, 777, 6, 0, 6, 10)
    assert st_seed == PREFIX_CACHE_OK

    st, slot, gen, tok = insert_or_update_preflight_only(c, 777, 6, 6, 6, 11)
    assert st == PREFIX_CACHE_OK
    assert slot == idx
    assert tok == 6
    assert gen == c.entries[idx].generation


def test_bad_params_rejected():
    c = make_cache(2)
    assert insert_or_update_preflight_only(c, -1, 8, 0, 8, 1)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert insert_or_update_preflight_only(c, 1, 8, 0, 8, -1)[0] == PREFIX_CACHE_ERR_BAD_PARAM


if __name__ == "__main__":
    test_new_insert_rejected_by_strict_parity()
    test_existing_update_passes_strict_parity()
    test_bad_params_rejected()
    print("ok")

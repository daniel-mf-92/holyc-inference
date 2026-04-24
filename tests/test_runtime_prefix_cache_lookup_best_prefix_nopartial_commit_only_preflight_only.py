#!/usr/bin/env python3
"""Host-side parity harness for IQ-1297.

Validates diagnostics semantics for
PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnly.
Runtime implementation remains HolyC-only.
"""

from dataclasses import dataclass

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1


@dataclass
class PrefixCacheEntry:
    valid: int = PREFIX_CACHE_FRESH_EMPTY
    prefix_hash: int = 0
    prefix_tokens: int = 0
    kv_start_token: int = 0
    kv_token_count: int = 0
    last_used_tick: int = 0


@dataclass
class PrefixCache:
    entries: list
    capacity: int
    count: int


def _lookup_best_prefix_nopartial(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    best_index = -1
    best_tokens = 0
    for idx, entry in enumerate(cache.entries):
        if entry.valid == PREFIX_CACHE_FRESH_VALID and entry.prefix_hash == query_hash:
            if entry.prefix_tokens <= max_prompt_tokens:
                if (
                    best_index < 0
                    or entry.prefix_tokens > best_tokens
                    or (entry.prefix_tokens == best_tokens and idx < best_index)
                ):
                    best_index = idx
                    best_tokens = entry.prefix_tokens

    if best_index < 0:
        return PREFIX_CACHE_ERR_NOT_FOUND, None, None

    return PREFIX_CACHE_OK, best_index, best_tokens


def _lookup_best_prefix_commit_only(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snapshot_entry_count = cache.count
    snapshot_max_prompt_tokens = max_prompt_tokens
    snapshot_query_hash = query_hash

    status, best_index, best_tokens = _lookup_best_prefix_nopartial(
        cache, query_hash, max_prompt_tokens
    )
    if status != PREFIX_CACHE_OK:
        return status, None, None

    if cache.count != snapshot_entry_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if max_prompt_tokens != snapshot_max_prompt_tokens or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash != snapshot_query_hash:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, best_index, best_tokens


def lookup_best_prefix_commit_only_preflight_only(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snapshot_entry_count = cache.count
    snapshot_max_prompt_tokens = max_prompt_tokens
    snapshot_query_hash = query_hash

    status_preflight, preflight_best_index, preflight_best_tokens = _lookup_best_prefix_commit_only(
        cache, query_hash, max_prompt_tokens
    )
    if status_preflight != PREFIX_CACHE_OK:
        return status_preflight, None, None

    status_canonical, canonical_best_index, canonical_best_tokens = _lookup_best_prefix_nopartial(
        cache, query_hash, max_prompt_tokens
    )
    if status_canonical != PREFIX_CACHE_OK:
        return status_canonical, None, None

    if preflight_best_index != canonical_best_index or preflight_best_tokens != canonical_best_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    if cache.count != snapshot_entry_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if max_prompt_tokens != snapshot_max_prompt_tokens or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash != snapshot_query_hash:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, preflight_best_index, preflight_best_tokens


def _build_cache(entries):
    valid_count = sum(1 for entry in entries if entry.valid == PREFIX_CACHE_FRESH_VALID)
    return PrefixCache(entries=entries, capacity=len(entries), count=valid_count)


def test_iq1297_success_and_tiebreak_lowest_index_wins():
    entries = [PrefixCacheEntry() for _ in range(6)]
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=77, prefix_tokens=48)
    entries[4] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=77, prefix_tokens=48)
    entries[5] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=77, prefix_tokens=32)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=13, prefix_tokens=64)

    cache = _build_cache(entries)
    before = [PrefixCacheEntry(**entry.__dict__) for entry in cache.entries]
    before_count = cache.count

    status, best_index, best_tokens = lookup_best_prefix_commit_only_preflight_only(cache, 77, 48)

    assert status == PREFIX_CACHE_OK
    assert (best_index, best_tokens) == (2, 48)
    assert cache.count == before_count
    assert [entry.__dict__ for entry in cache.entries] == [entry.__dict__ for entry in before]


def test_iq1297_enforces_max_prompt_tokens_bound():
    entries = [PrefixCacheEntry() for _ in range(5)]
    entries[0] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=8)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=24)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=32)

    cache = _build_cache(entries)
    status, best_index, best_tokens = lookup_best_prefix_commit_only_preflight_only(cache, 99, 20)

    assert status == PREFIX_CACHE_OK
    assert (best_index, best_tokens) == (0, 8)


def test_iq1297_not_found_returns_not_found_and_no_outputs():
    entries = [PrefixCacheEntry() for _ in range(4)]
    entries[3] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=12, prefix_tokens=10)
    cache = _build_cache(entries)

    status, best_index, best_tokens = lookup_best_prefix_commit_only_preflight_only(cache, 55, 100)

    assert status == PREFIX_CACHE_ERR_NOT_FOUND
    assert best_index is None
    assert best_tokens is None


def test_iq1297_bad_params_negative_hash_or_tokens_or_bad_capacity():
    cache = _build_cache([PrefixCacheEntry() for _ in range(2)])

    status, _, _ = lookup_best_prefix_commit_only_preflight_only(cache, -1, 8)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    status, _, _ = lookup_best_prefix_commit_only_preflight_only(cache, 1, -8)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM

    bad_capacity_cache = PrefixCache(entries=cache.entries, capacity=0, count=0)
    status, _, _ = lookup_best_prefix_commit_only_preflight_only(bad_capacity_cache, 1, 8)
    assert status == PREFIX_CACHE_ERR_BAD_PARAM


def test_iq1297_null_cache_or_entries_returns_null_ptr():
    status, _, _ = lookup_best_prefix_commit_only_preflight_only(None, 11, 7)
    assert status == PREFIX_CACHE_ERR_NULL_PTR

    cache = PrefixCache(entries=None, capacity=2, count=0)
    status, _, _ = lookup_best_prefix_commit_only_preflight_only(cache, 11, 7)
    assert status == PREFIX_CACHE_ERR_NULL_PTR

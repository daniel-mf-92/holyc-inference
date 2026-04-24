#!/usr/bin/env python3
"""Parity harness for PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1298)."""

from dataclasses import dataclass, field
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = -1
PREFIX_CACHE_ERR_BAD_PARAM = -2
PREFIX_CACHE_ERR_NOT_FOUND = -3

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1


@dataclass
class PrefixCacheEntry:
    valid: int = PREFIX_CACHE_FRESH_EMPTY
    prefix_hash: int = 0
    prefix_tokens: int = 0


@dataclass
class PrefixCache:
    entries: list[PrefixCacheEntry] | None
    capacity: int
    count: int = 0


def lookup_best_prefix_nopartial(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    best_index = None
    best_tokens = 0
    for idx in range(cache.capacity):
        entry = cache.entries[idx]
        if entry.valid == PREFIX_CACHE_FRESH_VALID and entry.prefix_hash == query_hash:
            if entry.prefix_tokens <= max_prompt_tokens:
                if (
                    best_index is None
                    or entry.prefix_tokens > best_tokens
                    or (entry.prefix_tokens == best_tokens and idx < best_index)
                ):
                    best_index = idx
                    best_tokens = entry.prefix_tokens

    if best_index is None:
        return PREFIX_CACHE_ERR_NOT_FOUND, None, None
    return PREFIX_CACHE_OK, best_index, best_tokens


def lookup_best_prefix_commit_only(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snapshot_count = cache.count
    snapshot_query_hash = query_hash
    snapshot_max_prompt_tokens = max_prompt_tokens

    status, best_index, best_tokens = lookup_best_prefix_nopartial(cache, query_hash, max_prompt_tokens)
    if status != PREFIX_CACHE_OK:
        return status, None, None

    if cache.count != snapshot_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash != snapshot_query_hash:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if max_prompt_tokens != snapshot_max_prompt_tokens or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, best_index, best_tokens


def lookup_best_prefix_commit_only_preflight_only(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snapshot_count = cache.count
    snapshot_query_hash = query_hash
    snapshot_max_prompt_tokens = max_prompt_tokens

    status_preflight, preflight_best_index, preflight_best_tokens = lookup_best_prefix_commit_only(
        cache, query_hash, max_prompt_tokens
    )
    if status_preflight != PREFIX_CACHE_OK:
        return status_preflight, None, None

    status_canonical, canonical_best_index, canonical_best_tokens = lookup_best_prefix_nopartial(
        cache, query_hash, max_prompt_tokens
    )
    if status_canonical != PREFIX_CACHE_OK:
        return status_canonical, None, None

    if preflight_best_index != canonical_best_index or preflight_best_tokens != canonical_best_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    if cache.count != snapshot_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash != snapshot_query_hash:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if max_prompt_tokens != snapshot_max_prompt_tokens or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, preflight_best_index, preflight_best_tokens


def lookup_best_prefix_commit_only_preflight_only_parity(cache, query_hash, max_prompt_tokens):
    if cache is None or cache.entries is None:
        return PREFIX_CACHE_ERR_NULL_PTR, None, None
    if cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash < 0 or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    snapshot_count = cache.count
    snapshot_capacity = cache.capacity
    snapshot_query_hash = query_hash
    snapshot_max_prompt_tokens = max_prompt_tokens

    status_preflight, preflight_best_index, preflight_best_tokens = lookup_best_prefix_commit_only_preflight_only(
        cache, query_hash, max_prompt_tokens
    )
    if status_preflight != PREFIX_CACHE_OK:
        return status_preflight, None, None

    status_canonical, canonical_best_index, canonical_best_tokens = lookup_best_prefix_commit_only(
        cache, query_hash, max_prompt_tokens
    )
    if status_canonical != PREFIX_CACHE_OK:
        return status_canonical, None, None

    if preflight_best_index != canonical_best_index or preflight_best_tokens != canonical_best_tokens:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    if cache.count != snapshot_count or cache.count < 0 or cache.count > cache.capacity:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if cache.capacity != snapshot_capacity or cache.capacity <= 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if query_hash != snapshot_query_hash:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None
    if max_prompt_tokens != snapshot_max_prompt_tokens or max_prompt_tokens < 0:
        return PREFIX_CACHE_ERR_BAD_PARAM, None, None

    return PREFIX_CACHE_OK, preflight_best_index, preflight_best_tokens


def test_parity_success_and_tiebreak_lowest_index():
    entries = [PrefixCacheEntry() for _ in range(7)]
    cache = PrefixCache(entries=entries, capacity=7, count=4)
    entries[4] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=48)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=48)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=32)

    status, best_index, best_tokens = lookup_best_prefix_commit_only_preflight_only_parity(cache, 99, 64)
    assert status == PREFIX_CACHE_OK
    assert best_index == 2
    assert best_tokens == 48


def test_parity_respects_prompt_bound():
    entries = [PrefixCacheEntry() for _ in range(5)]
    cache = PrefixCache(entries=entries, capacity=5, count=3)
    entries[0] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=7, prefix_tokens=8)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=7, prefix_tokens=24)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=7, prefix_tokens=32)

    status, best_index, best_tokens = lookup_best_prefix_commit_only_preflight_only_parity(cache, 7, 24)
    assert status == PREFIX_CACHE_OK
    assert best_index == 1
    assert best_tokens == 24


def test_parity_not_found_bubbles_without_write():
    entries = [PrefixCacheEntry() for _ in range(3)]
    cache = PrefixCache(entries=entries, capacity=3, count=1)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=123, prefix_tokens=10)

    status, best_index, best_tokens = lookup_best_prefix_commit_only_preflight_only_parity(cache, 124, 10)
    assert status == PREFIX_CACHE_ERR_NOT_FOUND
    assert best_index is None
    assert best_tokens is None
    assert cache.count == 1


def test_holyc_function_body_and_calls_present():
    source = Path("src/runtime/prefix_cache.HC").read_text(encoding="utf-8")
    sig = "I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly(" in body
    assert "cache->count != snapshot_entry_count" in body
    assert "cache->capacity != snapshot_capacity" in body
    assert "*out_best_index = preflight_best_index;" in body
    assert "*out_best_tokens = preflight_best_tokens;" in body


if __name__ == "__main__":
    test_parity_success_and_tiebreak_lowest_index()
    test_parity_respects_prompt_bound()
    test_parity_not_found_bubbles_without_write()
    test_holyc_function_body_and_calls_present()
    print("ok")

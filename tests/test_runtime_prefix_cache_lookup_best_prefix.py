#!/usr/bin/env python3
"""Host-side harness for IQ-1288 PrefixCacheLookupBestPrefixChecked."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREFIX_CACHE_OK = 0
PREFIX_CACHE_ERR_NULL_PTR = 1
PREFIX_CACHE_ERR_BAD_PARAM = 2
PREFIX_CACHE_ERR_NOT_FOUND = 4

PREFIX_CACHE_FRESH_EMPTY = 0
PREFIX_CACHE_FRESH_VALID = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "src" / "runtime" / "prefix_cache.HC"


@dataclass
class PrefixCacheEntry:
    valid: int = PREFIX_CACHE_FRESH_EMPTY
    prefix_hash: int = 0
    prefix_tokens: int = 0


@dataclass
class PrefixCache:
    entries: list[PrefixCacheEntry] | None
    capacity: int


def prefix_cache_lookup_best_prefix_checked(
    cache: PrefixCache | None,
    query_hash: int,
    max_prompt_tokens: int,
):
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


def test_iq1288_selects_longest_prefix_under_bound():
    entries = [PrefixCacheEntry() for _ in range(5)]
    entries[0] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=42, prefix_tokens=8)
    entries[1] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=42, prefix_tokens=24)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=42, prefix_tokens=40)
    entries[3] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=99, prefix_tokens=40)

    status, best_index, best_tokens = prefix_cache_lookup_best_prefix_checked(
        PrefixCache(entries=entries, capacity=len(entries)),
        42,
        24,
    )
    assert status == PREFIX_CACHE_OK
    assert (best_index, best_tokens) == (1, 24)


def test_iq1288_stable_tiebreak_prefers_lowest_index():
    entries = [PrefixCacheEntry() for _ in range(6)]
    entries[4] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=7, prefix_tokens=32)
    entries[2] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=7, prefix_tokens=32)

    status, best_index, best_tokens = prefix_cache_lookup_best_prefix_checked(
        PrefixCache(entries=entries, capacity=len(entries)),
        7,
        32,
    )
    assert status == PREFIX_CACHE_OK
    assert (best_index, best_tokens) == (2, 32)


def test_iq1288_not_found_and_bad_params():
    entries = [PrefixCacheEntry() for _ in range(2)]
    entries[0] = PrefixCacheEntry(valid=PREFIX_CACHE_FRESH_VALID, prefix_hash=3, prefix_tokens=4)
    cache = PrefixCache(entries=entries, capacity=2)

    status, best_index, best_tokens = prefix_cache_lookup_best_prefix_checked(cache, 8, 100)
    assert status == PREFIX_CACHE_ERR_NOT_FOUND
    assert best_index is None
    assert best_tokens is None

    assert prefix_cache_lookup_best_prefix_checked(None, 8, 10)[0] == PREFIX_CACHE_ERR_NULL_PTR
    assert prefix_cache_lookup_best_prefix_checked(cache, -1, 10)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert prefix_cache_lookup_best_prefix_checked(cache, 1, -10)[0] == PREFIX_CACHE_ERR_BAD_PARAM
    assert (
        prefix_cache_lookup_best_prefix_checked(PrefixCache(entries=entries, capacity=0), 1, 10)[0]
        == PREFIX_CACHE_ERR_BAD_PARAM
    )


def test_iq1288_source_contains_function_and_stable_tiebreak():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    signature = (
        "I32 PrefixCacheLookupBestPrefixChecked(PrefixCache *cache,\n"
        "                                       I64 query_hash,\n"
        "                                       I64 max_prompt_tokens,\n"
        "                                       I64 *out_best_index,\n"
        "                                       I64 *out_best_tokens)\n"
    )
    assert signature in source
    assert "(prefix_tokens == best_tokens && idx < best_index)" in source


if __name__ == "__main__":
    test_iq1288_selects_longest_prefix_under_bound()
    test_iq1288_stable_tiebreak_prefers_lowest_index()
    test_iq1288_not_found_and_bad_params()
    test_iq1288_source_contains_function_and_stable_tiebreak()
    print("ok")

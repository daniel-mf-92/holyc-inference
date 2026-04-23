import ctypes
import subprocess
import tempfile
from pathlib import Path


def compile_lib() -> Path:
    src = r'''
    #include <stdint.h>

    #define PREFIX_CACHE_OK            0
    #define PREFIX_CACHE_ERR_NULL_PTR  1
    #define PREFIX_CACHE_ERR_BAD_PARAM 2
    #define PREFIX_CACHE_ERR_FULL      3
    #define PREFIX_CACHE_ERR_NOT_FOUND 4

    #define PREFIX_CACHE_FRESH_EMPTY 0
    #define PREFIX_CACHE_FRESH_VALID 1

    typedef struct PrefixCacheEntry {
        int64_t valid;
        int64_t prefix_hash;
        int64_t prefix_tokens;
        int64_t kv_start_token;
        int64_t kv_token_count;
        int64_t last_used_tick;
    } PrefixCacheEntry;

    typedef struct PrefixCache {
        PrefixCacheEntry *entries;
        int64_t capacity;
        int64_t count;
    } PrefixCache;

    int32_t PrefixCacheInitChecked(PrefixCache *cache,
                                   PrefixCacheEntry *entry_buffer,
                                   int64_t entry_capacity)
    {
        int64_t idx;
        if (!cache || !entry_buffer) return PREFIX_CACHE_ERR_NULL_PTR;
        if (entry_capacity <= 0) return PREFIX_CACHE_ERR_BAD_PARAM;
        cache->entries = entry_buffer;
        cache->capacity = entry_capacity;
        cache->count = 0;
        for (idx = 0; idx < entry_capacity; idx++) {
            cache->entries[idx].valid = PREFIX_CACHE_FRESH_EMPTY;
            cache->entries[idx].prefix_hash = 0;
            cache->entries[idx].prefix_tokens = 0;
            cache->entries[idx].kv_start_token = 0;
            cache->entries[idx].kv_token_count = 0;
            cache->entries[idx].last_used_tick = 0;
        }
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheFindIndexChecked(PrefixCache *cache,
                                        int64_t prefix_hash,
                                        int64_t prefix_tokens,
                                        int64_t *out_index)
    {
        int64_t idx;
        if (!cache || !cache->entries || !out_index) return PREFIX_CACHE_ERR_NULL_PTR;
        if (cache->capacity <= 0) return PREFIX_CACHE_ERR_BAD_PARAM;
        if (prefix_hash < 0 || prefix_tokens < 0) return PREFIX_CACHE_ERR_BAD_PARAM;
        for (idx = 0; idx < cache->capacity; idx++) {
            if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID &&
                cache->entries[idx].prefix_hash == prefix_hash &&
                cache->entries[idx].prefix_tokens == prefix_tokens) {
                *out_index = idx;
                return PREFIX_CACHE_OK;
            }
        }
        return PREFIX_CACHE_ERR_NOT_FOUND;
    }

    int32_t PrefixCacheSelectVictimIndexLRUChecked(PrefixCache *cache,
                                                   int64_t *out_victim_index)
    {
        int64_t idx;
        int64_t oldest_tick;
        int64_t victim_index;
        if (!cache || !cache->entries || !out_victim_index) return PREFIX_CACHE_ERR_NULL_PTR;
        if (cache->capacity <= 0) return PREFIX_CACHE_ERR_BAD_PARAM;

        for (idx = 0; idx < cache->capacity; idx++) {
            if (cache->entries[idx].valid != PREFIX_CACHE_FRESH_VALID) {
                *out_victim_index = idx;
                return PREFIX_CACHE_OK;
            }
        }

        oldest_tick = cache->entries[0].last_used_tick;
        victim_index = 0;
        for (idx = 1; idx < cache->capacity; idx++) {
            if (cache->entries[idx].last_used_tick < oldest_tick) {
                oldest_tick = cache->entries[idx].last_used_tick;
                victim_index = idx;
            }
        }
        *out_victim_index = victim_index;
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheInsertOrUpdateChecked(PrefixCache *cache,
                                             int64_t prefix_hash,
                                             int64_t prefix_tokens,
                                             int64_t kv_start_token,
                                             int64_t kv_token_count,
                                             int64_t access_tick,
                                             int64_t *out_entry_index,
                                             int64_t *out_inserted_new)
    {
        int64_t found_index;
        int64_t victim_index;
        int32_t status;

        if (!cache || !cache->entries || !out_entry_index || !out_inserted_new)
            return PREFIX_CACHE_ERR_NULL_PTR;

        if (cache->capacity <= 0 || access_tick < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        if (prefix_hash < 0 || prefix_tokens < 0 || kv_start_token < 0 || kv_token_count < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        status = PrefixCacheFindIndexChecked(cache, prefix_hash, prefix_tokens, &found_index);
        if (status == PREFIX_CACHE_OK) {
            cache->entries[found_index].kv_start_token = kv_start_token;
            cache->entries[found_index].kv_token_count = kv_token_count;
            cache->entries[found_index].last_used_tick = access_tick;
            *out_entry_index = found_index;
            *out_inserted_new = 0;
            return PREFIX_CACHE_OK;
        }

        if (status != PREFIX_CACHE_ERR_NOT_FOUND) return status;

        status = PrefixCacheSelectVictimIndexLRUChecked(cache, &victim_index);
        if (status != PREFIX_CACHE_OK) return status;

        if (cache->entries[victim_index].valid != PREFIX_CACHE_FRESH_VALID) cache->count++;

        cache->entries[victim_index].valid = PREFIX_CACHE_FRESH_VALID;
        cache->entries[victim_index].prefix_hash = prefix_hash;
        cache->entries[victim_index].prefix_tokens = prefix_tokens;
        cache->entries[victim_index].kv_start_token = kv_start_token;
        cache->entries[victim_index].kv_token_count = kv_token_count;
        cache->entries[victim_index].last_used_tick = access_tick;

        *out_entry_index = victim_index;
        *out_inserted_new = 1;
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheLookupBestPrefixCheckedNoPartial(PrefixCache *cache,
                                                        int64_t query_hash,
                                                        int64_t max_prompt_tokens,
                                                        int64_t *out_best_index,
                                                        int64_t *out_best_tokens)
    {
        int64_t idx;
        int64_t best_index;
        int64_t best_tokens;
        int64_t prefix_tokens;

        if (!cache || !cache->entries || !out_best_index || !out_best_tokens)
            return PREFIX_CACHE_ERR_NULL_PTR;
        if (out_best_index == out_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (cache->capacity <= 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash < 0 || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        best_index = -1;
        best_tokens = 0;

        for (idx = 0; idx < cache->capacity; idx++) {
            if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID &&
                cache->entries[idx].prefix_hash == query_hash) {
                prefix_tokens = cache->entries[idx].prefix_tokens;
                if (prefix_tokens <= max_prompt_tokens) {
                    if (best_index < 0 || prefix_tokens > best_tokens ||
                        (prefix_tokens == best_tokens && idx < best_index)) {
                        best_index = idx;
                        best_tokens = prefix_tokens;
                    }
                }
            }
        }

        if (best_index < 0)
            return PREFIX_CACHE_ERR_NOT_FOUND;

        *out_best_index = best_index;
        *out_best_tokens = best_tokens;
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly(PrefixCache *cache,
                                                                  int64_t query_hash,
                                                                  int64_t max_prompt_tokens,
                                                                  int64_t *out_best_index,
                                                                  int64_t *out_best_tokens)
    {
        int64_t snapshot_entry_count;
        int64_t snapshot_max_prompt_tokens;
        int64_t snapshot_query_hash;
        int64_t best_index;
        int64_t best_tokens;
        int32_t status;

        if (!cache || !cache->entries || !out_best_index || !out_best_tokens)
            return PREFIX_CACHE_ERR_NULL_PTR;
        if (out_best_index == out_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (cache->capacity <= 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash < 0 || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        snapshot_entry_count = cache->count;
        snapshot_max_prompt_tokens = max_prompt_tokens;
        snapshot_query_hash = query_hash;

        status = PrefixCacheLookupBestPrefixCheckedNoPartial(cache,
                                                             query_hash,
                                                             max_prompt_tokens,
                                                             &best_index,
                                                             &best_tokens);
        if (status != PREFIX_CACHE_OK)
            return status;

        if (cache->count != snapshot_entry_count || cache->count < 0 || cache->count > cache->capacity)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (max_prompt_tokens != snapshot_max_prompt_tokens || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash != snapshot_query_hash)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        *out_best_index = best_index;
        *out_best_tokens = best_tokens;
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                                int64_t query_hash,
                                                                                int64_t max_prompt_tokens,
                                                                                int64_t *out_best_index,
                                                                                int64_t *out_best_tokens)
    {
        int64_t snapshot_entry_count;
        int64_t snapshot_max_prompt_tokens;
        int64_t snapshot_query_hash;
        int64_t preflight_best_index;
        int64_t preflight_best_tokens;
        int64_t canonical_best_index;
        int64_t canonical_best_tokens;
        int32_t status_preflight;
        int32_t status_canonical;

        if (!cache || !cache->entries || !out_best_index || !out_best_tokens)
            return PREFIX_CACHE_ERR_NULL_PTR;
        if (out_best_index == out_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (cache->capacity <= 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash < 0 || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        snapshot_entry_count = cache->count;
        snapshot_max_prompt_tokens = max_prompt_tokens;
        snapshot_query_hash = query_hash;

        status_preflight = PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly(cache,
                                                                                  query_hash,
                                                                                  max_prompt_tokens,
                                                                                  &preflight_best_index,
                                                                                  &preflight_best_tokens);
        if (status_preflight != PREFIX_CACHE_OK)
            return status_preflight;

        status_canonical = PrefixCacheLookupBestPrefixCheckedNoPartial(cache,
                                                                       query_hash,
                                                                       max_prompt_tokens,
                                                                       &canonical_best_index,
                                                                       &canonical_best_tokens);
        if (status_canonical != PREFIX_CACHE_OK)
            return status_canonical;

        if (preflight_best_index != canonical_best_index ||
            preflight_best_tokens != canonical_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        if (cache->count != snapshot_entry_count || cache->count < 0 || cache->count > cache->capacity)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (max_prompt_tokens != snapshot_max_prompt_tokens || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash != snapshot_query_hash)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        *out_best_index = preflight_best_index;
        *out_best_tokens = preflight_best_tokens;
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                                      int64_t query_hash,
                                                                                      int64_t max_prompt_tokens,
                                                                                      int64_t *out_best_index,
                                                                                      int64_t *out_best_tokens)
    {
        int64_t snapshot_entry_count;
        int64_t snapshot_max_prompt_tokens;
        int64_t snapshot_query_hash;
        int64_t preflight_best_index;
        int64_t preflight_best_tokens;
        int64_t canonical_best_index;
        int64_t canonical_best_tokens;
        int32_t status_preflight;
        int32_t status_canonical;

        if (!cache || !cache->entries || !out_best_index || !out_best_tokens)
            return PREFIX_CACHE_ERR_NULL_PTR;
        if (out_best_index == out_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (cache->capacity <= 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash < 0 || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        snapshot_entry_count = cache->count;
        snapshot_max_prompt_tokens = max_prompt_tokens;
        snapshot_query_hash = query_hash;

        status_preflight = PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnly(cache,
                                                                                               query_hash,
                                                                                               max_prompt_tokens,
                                                                                               &preflight_best_index,
                                                                                               &preflight_best_tokens);
        if (status_preflight != PREFIX_CACHE_OK)
            return status_preflight;

        status_canonical = PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly(cache,
                                                                                 query_hash,
                                                                                 max_prompt_tokens,
                                                                                 &canonical_best_index,
                                                                                 &canonical_best_tokens);
        if (status_canonical != PREFIX_CACHE_OK)
            return status_canonical;

        if (preflight_best_index != canonical_best_index ||
            preflight_best_tokens != canonical_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        if (cache->count != snapshot_entry_count || cache->count < 0 || cache->count > cache->capacity)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (max_prompt_tokens != snapshot_max_prompt_tokens || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash != snapshot_query_hash)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        *out_best_index = preflight_best_index;
        *out_best_tokens = preflight_best_tokens;
        return PREFIX_CACHE_OK;
    }

    int32_t PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,
                                                                                                 int64_t query_hash,
                                                                                                 int64_t max_prompt_tokens,
                                                                                                 int64_t *out_best_index,
                                                                                                 int64_t *out_best_tokens)
    {
        int64_t snapshot_entry_count;
        int64_t snapshot_max_prompt_tokens;
        int64_t snapshot_query_hash;
        int64_t parity_best_index;
        int64_t parity_best_tokens;
        int64_t canonical_best_index;
        int64_t canonical_best_tokens;
        int32_t status_parity;
        int32_t status_canonical;

        if (!cache || !cache->entries || !out_best_index || !out_best_tokens)
            return PREFIX_CACHE_ERR_NULL_PTR;
        if (out_best_index == out_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (cache->capacity <= 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash < 0 || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        snapshot_entry_count = cache->count;
        snapshot_max_prompt_tokens = max_prompt_tokens;
        snapshot_query_hash = query_hash;

        status_parity = PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParity(cache,
                                                                                                   query_hash,
                                                                                                   max_prompt_tokens,
                                                                                                   &parity_best_index,
                                                                                                   &parity_best_tokens);
        if (status_parity != PREFIX_CACHE_OK)
            return status_parity;

        status_canonical = PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly(cache,
                                                                                  query_hash,
                                                                                  max_prompt_tokens,
                                                                                  &canonical_best_index,
                                                                                  &canonical_best_tokens);
        if (status_canonical != PREFIX_CACHE_OK)
            return status_canonical;

        if (parity_best_index != canonical_best_index || parity_best_tokens != canonical_best_tokens)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (cache->count != snapshot_entry_count || cache->count < 0 || cache->count > cache->capacity)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (max_prompt_tokens != snapshot_max_prompt_tokens || max_prompt_tokens < 0)
            return PREFIX_CACHE_ERR_BAD_PARAM;
        if (query_hash != snapshot_query_hash)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        *out_best_index = parity_best_index;
        *out_best_tokens = parity_best_tokens;
        return PREFIX_CACHE_OK;
    }
    '''
    td = Path(tempfile.mkdtemp())
    c_path = td / "prefix_cache_parity_commit_only.c"
    so_path = td / "libprefix_cache_parity_commit_only.so"
    c_path.write_text(src)
    subprocess.run([
        "cc", "-shared", "-fPIC", "-O2", str(c_path), "-o", str(so_path)
    ], check=True)
    return so_path


def test_lookup_best_prefix_nopartial_preflight_parity_commit_only():
    so = ctypes.CDLL(str(compile_lib()))

    class PrefixCacheEntry(ctypes.Structure):
        _fields_ = [
            ("valid", ctypes.c_int64),
            ("prefix_hash", ctypes.c_int64),
            ("prefix_tokens", ctypes.c_int64),
            ("kv_start_token", ctypes.c_int64),
            ("kv_token_count", ctypes.c_int64),
            ("last_used_tick", ctypes.c_int64),
        ]

    class PrefixCache(ctypes.Structure):
        _fields_ = [
            ("entries", ctypes.POINTER(PrefixCacheEntry)),
            ("capacity", ctypes.c_int64),
            ("count", ctypes.c_int64),
        ]

    init = so.PrefixCacheInitChecked
    init.argtypes = [ctypes.POINTER(PrefixCache), ctypes.POINTER(PrefixCacheEntry), ctypes.c_int64]
    init.restype = ctypes.c_int32

    insert = so.PrefixCacheInsertOrUpdateChecked
    insert.argtypes = [
        ctypes.POINTER(PrefixCache), ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)
    ]
    insert.restype = ctypes.c_int32

    lookup = so.PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly
    lookup.argtypes = [
        ctypes.POINTER(PrefixCache), ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)
    ]
    lookup.restype = ctypes.c_int32

    entries = (PrefixCacheEntry * 8)()
    cache = PrefixCache(entries, 0, 0)
    assert init(ctypes.byref(cache), entries, 8) == 0

    slot = ctypes.c_int64()
    is_new = ctypes.c_int64()

    assert insert(ctypes.byref(cache), 777, 4, 0, 4, 10, ctypes.byref(slot), ctypes.byref(is_new)) == 0
    assert insert(ctypes.byref(cache), 777, 8, 0, 8, 11, ctypes.byref(slot), ctypes.byref(is_new)) == 0
    assert insert(ctypes.byref(cache), 777, 12, 0, 12, 12, ctypes.byref(slot), ctypes.byref(is_new)) == 0
    assert insert(ctypes.byref(cache), 778, 16, 0, 16, 13, ctypes.byref(slot), ctypes.byref(is_new)) == 0

    count_before = cache.count
    best_idx = ctypes.c_int64(-1)
    best_toks = ctypes.c_int64(-1)

    status = lookup(ctypes.byref(cache), 777, 9, ctypes.byref(best_idx), ctypes.byref(best_toks))
    assert status == 0
    assert best_toks.value == 8
    assert best_idx.value >= 0
    assert cache.count == count_before


def test_lookup_best_prefix_nopartial_preflight_parity_commit_only_not_found_and_bad_params():
    so = ctypes.CDLL(str(compile_lib()))

    class PrefixCacheEntry(ctypes.Structure):
        _fields_ = [
            ("valid", ctypes.c_int64),
            ("prefix_hash", ctypes.c_int64),
            ("prefix_tokens", ctypes.c_int64),
            ("kv_start_token", ctypes.c_int64),
            ("kv_token_count", ctypes.c_int64),
            ("last_used_tick", ctypes.c_int64),
        ]

    class PrefixCache(ctypes.Structure):
        _fields_ = [
            ("entries", ctypes.POINTER(PrefixCacheEntry)),
            ("capacity", ctypes.c_int64),
            ("count", ctypes.c_int64),
        ]

    init = so.PrefixCacheInitChecked
    init.argtypes = [ctypes.POINTER(PrefixCache), ctypes.POINTER(PrefixCacheEntry), ctypes.c_int64]
    init.restype = ctypes.c_int32

    lookup = so.PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly
    lookup.argtypes = [
        ctypes.POINTER(PrefixCache), ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)
    ]
    lookup.restype = ctypes.c_int32

    entries = (PrefixCacheEntry * 4)()
    cache = PrefixCache(entries, 0, 0)
    assert init(ctypes.byref(cache), entries, 4) == 0

    out_idx = ctypes.c_int64()
    out_toks = ctypes.c_int64()

    assert lookup(ctypes.byref(cache), 888, 10, ctypes.byref(out_idx), ctypes.byref(out_toks)) == 4
    assert lookup(ctypes.byref(cache), -1, 10, ctypes.byref(out_idx), ctypes.byref(out_toks)) == 2
    assert lookup(ctypes.byref(cache), 888, -1, ctypes.byref(out_idx), ctypes.byref(out_toks)) == 2

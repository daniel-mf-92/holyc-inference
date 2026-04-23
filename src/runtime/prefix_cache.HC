// prefix_cache.HC
// Integer-only prefix cache manager for inference runtime.
//
// Design: direct, auditable array-of-records cache with no allocations.
// Each entry binds a prompt token prefix hash to a KV cache window snapshot.
//
// Notes:
// - secure-local/dev-local policy is not bypassed here; this module is pure data-path.
// - caller is responsible for profile gates + attestation before enabling reuse.

#define PREFIX_CACHE_OK            0
#define PREFIX_CACHE_ERR_NULL_PTR  1
#define PREFIX_CACHE_ERR_BAD_PARAM 2
#define PREFIX_CACHE_ERR_FULL      3
#define PREFIX_CACHE_ERR_NOT_FOUND 4

#define PREFIX_CACHE_FRESH_EMPTY 0
#define PREFIX_CACHE_FRESH_VALID 1

class PrefixCacheEntry
{
    I64 valid;
    I64 prefix_hash;
    I64 prefix_tokens;
    I64 kv_start_token;
    I64 kv_token_count;
    I64 last_used_tick;
};

class PrefixCache
{
    PrefixCacheEntry *entries;
    I64 capacity;
    I64 count;
};

I32 InferencePromptPrefixHashQ64Checked(U8 *tokens,
                                        I64 token_count,
                                        I64 *out_prefix_hash)
{
    I64 idx;
    I64 hash;
    I64 value;

    if (!tokens || !out_prefix_hash)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    hash = 1469598103934665603;
    idx = 0;
    while (idx < token_count)
    {
        value = tokens[idx];
        hash = hash ^ value;
        hash = hash * 1099511628211;
        idx++;
    }

    hash = hash ^ token_count;
    hash = hash * 1099511628211;
    hash = hash & 0x7FFFFFFFFFFFFFFF;

    *out_prefix_hash = hash;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInitChecked(PrefixCache *cache,
                           PrefixCacheEntry *entry_buffer,
                           I64 entry_capacity)
{
    I64 idx;

    if (!cache || !entry_buffer)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (entry_capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    cache->entries = entry_buffer;
    cache->capacity = entry_capacity;
    cache->count = 0;

    idx = 0;
    while (idx < entry_capacity)
    {
        cache->entries[idx].valid = PREFIX_CACHE_FRESH_EMPTY;
        cache->entries[idx].prefix_hash = 0;
        cache->entries[idx].prefix_tokens = 0;
        cache->entries[idx].kv_start_token = 0;
        cache->entries[idx].kv_token_count = 0;
        cache->entries[idx].last_used_tick = 0;
        idx++;
    }

    return PREFIX_CACHE_OK;
}

I32 PrefixCacheClearChecked(PrefixCache *cache)
{
    I64 idx;

    if (!cache || !cache->entries)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < cache->capacity)
    {
        cache->entries[idx].valid = PREFIX_CACHE_FRESH_EMPTY;
        cache->entries[idx].prefix_hash = 0;
        cache->entries[idx].prefix_tokens = 0;
        cache->entries[idx].kv_start_token = 0;
        cache->entries[idx].kv_token_count = 0;
        cache->entries[idx].last_used_tick = 0;
        idx++;
    }

    cache->count = 0;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheValidateTupleChecked(I64 prefix_hash,
                                    I64 prefix_tokens,
                                    I64 kv_start_token,
                                    I64 kv_token_count)
{
    if (prefix_hash < 0 ||
        prefix_tokens < 0 ||
        kv_start_token < 0 ||
        kv_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    return PREFIX_CACHE_OK;
}

I32 PrefixCacheFindIndexChecked(PrefixCache *cache,
                                I64 prefix_hash,
                                I64 prefix_tokens,
                                I64 *out_index)
{
    I64 idx;

    if (!cache || !cache->entries || !out_index)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (prefix_hash < 0 || prefix_tokens < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < cache->capacity)
    {
        if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID &&
            cache->entries[idx].prefix_hash == prefix_hash &&
            cache->entries[idx].prefix_tokens == prefix_tokens)
        {
            *out_index = idx;
            return PREFIX_CACHE_OK;
        }
        idx++;
    }

    return PREFIX_CACHE_ERR_NOT_FOUND;
}

I32 PrefixCacheLookupChecked(PrefixCache *cache,
                             I64 prefix_hash,
                             I64 prefix_tokens,
                             I64 access_tick,
                             I64 *out_kv_start_token,
                             I64 *out_kv_token_count,
                             I64 *out_entry_index)
{
    I64 found_index;
    I32 status;

    if (!out_kv_start_token || !out_kv_token_count || !out_entry_index)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status = PrefixCacheFindIndexChecked(cache,
                                         prefix_hash,
                                         prefix_tokens,
                                         &found_index);
    if (status != PREFIX_CACHE_OK)
        return status;

    cache->entries[found_index].last_used_tick = access_tick;

    *out_kv_start_token = cache->entries[found_index].kv_start_token;
    *out_kv_token_count = cache->entries[found_index].kv_token_count;
    *out_entry_index = found_index;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheSelectVictimIndexLRUChecked(PrefixCache *cache,
                                           I64 *out_victim_index)
{
    I64 idx;
    I64 oldest_tick;
    I64 victim_index;

    if (!cache || !cache->entries || !out_victim_index)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < cache->capacity)
    {
        if (cache->entries[idx].valid != PREFIX_CACHE_FRESH_VALID)
        {
            *out_victim_index = idx;
            return PREFIX_CACHE_OK;
        }
        idx++;
    }

    oldest_tick = cache->entries[0].last_used_tick;
    victim_index = 0;

    idx = 1;
    while (idx < cache->capacity)
    {
        if (cache->entries[idx].last_used_tick < oldest_tick)
        {
            oldest_tick = cache->entries[idx].last_used_tick;
            victim_index = idx;
        }
        idx++;
    }

    *out_victim_index = victim_index;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInsertOrUpdateChecked(PrefixCache *cache,
                                     I64 prefix_hash,
                                     I64 prefix_tokens,
                                     I64 kv_start_token,
                                     I64 kv_token_count,
                                     I64 access_tick,
                                     I64 *out_entry_index,
                                     I64 *out_inserted_new)
{
    I64 found_index;
    I64 victim_index;
    I32 status;

    if (!cache || !cache->entries || !out_entry_index || !out_inserted_new)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status = PrefixCacheValidateTupleChecked(prefix_hash,
                                             prefix_tokens,
                                             kv_start_token,
                                             kv_token_count);
    if (status != PREFIX_CACHE_OK)
        return status;

    status = PrefixCacheFindIndexChecked(cache,
                                         prefix_hash,
                                         prefix_tokens,
                                         &found_index);
    if (status == PREFIX_CACHE_OK)
    {
        cache->entries[found_index].kv_start_token = kv_start_token;
        cache->entries[found_index].kv_token_count = kv_token_count;
        cache->entries[found_index].last_used_tick = access_tick;

        *out_entry_index = found_index;
        *out_inserted_new = 0;
        return PREFIX_CACHE_OK;
    }

    if (status != PREFIX_CACHE_ERR_NOT_FOUND)
        return status;

    status = PrefixCacheSelectVictimIndexLRUChecked(cache, &victim_index);
    if (status != PREFIX_CACHE_OK)
        return status;

    if (cache->entries[victim_index].valid != PREFIX_CACHE_FRESH_VALID)
        cache->count++;

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

I32 PrefixCacheLookupBestPrefixCheckedNoPartial(PrefixCache *cache,
                                                I64 query_hash,
                                                I64 max_prompt_tokens,
                                                I64 *out_best_index,
                                                I64 *out_best_tokens)
{
    I64 idx;
    I64 best_index;
    I64 best_tokens;
    I64 prefix_tokens;

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

    idx = 0;
    while (idx < cache->capacity)
    {
        if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID &&
            cache->entries[idx].prefix_hash == query_hash)
        {
            prefix_tokens = cache->entries[idx].prefix_tokens;
            if (prefix_tokens <= max_prompt_tokens)
            {
                if (best_index < 0 ||
                    prefix_tokens > best_tokens ||
                    (prefix_tokens == best_tokens && idx < best_index))
                {
                    best_index = idx;
                    best_tokens = prefix_tokens;
                }
            }
        }
        idx++;
    }

    if (best_index < 0)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    *out_best_index = best_index;
    *out_best_tokens = best_tokens;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly(PrefixCache *cache,
                                                          I64 query_hash,
                                                          I64 max_prompt_tokens,
                                                          I64 *out_best_index,
                                                          I64 *out_best_tokens)
{
    I64 snapshot_entry_count;
    I64 snapshot_max_prompt_tokens;
    I64 snapshot_query_hash;
    I64 best_index;
    I64 best_tokens;
    I32 status;

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

    if (cache->count != snapshot_entry_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (max_prompt_tokens != snapshot_max_prompt_tokens ||
        max_prompt_tokens < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (query_hash != snapshot_query_hash)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_best_index = best_index;
    *out_best_tokens = best_tokens;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                        I64 query_hash,
                                                                        I64 max_prompt_tokens,
                                                                        I64 *out_best_index,
                                                                        I64 *out_best_tokens)
{
    I64 snapshot_entry_count;
    I64 snapshot_max_prompt_tokens;
    I64 snapshot_query_hash;
    I64 preflight_best_index;
    I64 preflight_best_tokens;
    I64 canonical_best_index;
    I64 canonical_best_tokens;
    I32 status_preflight;
    I32 status_canonical;

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

    if (cache->count != snapshot_entry_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (max_prompt_tokens != snapshot_max_prompt_tokens ||
        max_prompt_tokens < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (query_hash != snapshot_query_hash)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_best_index = preflight_best_index;
    *out_best_tokens = preflight_best_tokens;
    return PREFIX_CACHE_OK;
}

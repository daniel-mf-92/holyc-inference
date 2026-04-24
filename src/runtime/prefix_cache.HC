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
#define PREFIX_CACHE_ERR_POLICY    5

#define PREFIX_CACHE_PROFILE_SECURE_LOCAL 1
#define PREFIX_CACHE_PROFILE_DEV_LOCAL    2

#define PREFIX_CACHE_FRESH_EMPTY 0
#define PREFIX_CACHE_FRESH_VALID 1

#define PREFIX_CACHE_HASH_FNV_OFFSET_BASIS 1469598103934665603
#define PREFIX_CACHE_HASH_FNV_PRIME        1099511628211
#define PREFIX_CACHE_HASH_DOMAIN_TAG       0x5052454649584841
#define PREFIX_CACHE_AUDIT_TUPLE_WIDTH     7
#define PREFIX_CACHE_AUDIT_DOMAIN_TAG      0x5052464155444954

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
    U64 hash_u64;
    U64 token_lane;
    U64 token_count_u64;

    if (!tokens || !out_prefix_hash)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Deterministic 64-bit FNV-1a stream seeded with a domain tag so
    // prompt-prefix hashes cannot alias other runtime digests by construction.
    hash_u64 = PREFIX_CACHE_HASH_FNV_OFFSET_BASIS;
    hash_u64 ^= (U64)PREFIX_CACHE_HASH_DOMAIN_TAG;
    hash_u64 *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;

    idx = 0;
    while (idx < token_count)
    {
        token_lane = (U64)tokens[idx];
        hash_u64 ^= token_lane;
        hash_u64 *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
        idx++;
    }

    // Length-mix uses low/high halves + full count lane with a delimiter byte.
    // This keeps integer-only behavior and avoids signed-overflow semantics.
    token_count_u64 = (U64)token_count;
    hash_u64 ^= (U64)0xFF;
    hash_u64 *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    hash_u64 ^= (token_count_u64 & (U64)0xFFFFFFFF);
    hash_u64 *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    hash_u64 ^= (token_count_u64 >> 32);
    hash_u64 *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    hash_u64 ^= token_count_u64;
    hash_u64 *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;

    *out_prefix_hash = (I64)(hash_u64 & (U64)0x7FFFFFFFFFFFFFFF);
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

I32 PrefixCacheReplayGuardChecked(PrefixCache *cache,
                                  I64 entry_index,
                                  I64 access_tick,
                                  I64 profile_id,
                                  I64 *out_previous_tick,
                                  I64 *out_applied_tick,
                                  I64 *out_guard_passed)
{
    I64 previous_tick;
    I64 staged_applied_tick;
    I64 staged_guard_passed;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (profile_id != PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        profile_id != PREFIX_CACHE_PROFILE_DEV_LOCAL)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    previous_tick = cache->entries[entry_index].last_used_tick;
    if (previous_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // secure-local policy: ticks must be monotonic to prevent replay rollback.
    if (profile_id == PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        access_tick < previous_tick)
        return PREFIX_CACHE_ERR_POLICY;

    staged_applied_tick = access_tick;
    staged_guard_passed = 1;

    cache->entries[entry_index].last_used_tick = staged_applied_tick;

    *out_previous_tick = previous_tick;
    *out_applied_tick = staged_applied_tick;
    *out_guard_passed = staged_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnly(PrefixCache *cache,
                                                     I64 entry_index,
                                                     I64 access_tick,
                                                     I64 profile_id,
                                                     I64 *out_previous_tick,
                                                     I64 *out_applied_tick,
                                                     I64 *out_guard_passed)
{
    I64 snapshot_entry_index;
    I64 snapshot_access_tick;
    I64 snapshot_profile_id;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 staged_previous_tick;
    I64 staged_applied_tick;
    I64 staged_guard_passed;
    I32 status;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_index = entry_index;
    snapshot_access_tick = access_tick;
    snapshot_profile_id = profile_id;
    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;

    status = PrefixCacheReplayGuardChecked(cache,
                                           entry_index,
                                           access_tick,
                                           profile_id,
                                           &staged_previous_tick,
                                           &staged_applied_tick,
                                           &staged_guard_passed);
    if (status != PREFIX_CACHE_OK)
        return status;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (staged_previous_tick < 0 ||
        staged_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (staged_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != staged_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (staged_guard_passed != 0 && staged_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_previous_tick = staged_previous_tick;
    *out_applied_tick = staged_applied_tick;
    *out_guard_passed = staged_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                  I64 entry_index,
                                                                  I64 access_tick,
                                                                  I64 profile_id,
                                                                  I64 *out_previous_tick,
                                                                  I64 *out_applied_tick,
                                                                  I64 *out_guard_passed)
{
    I64 snapshot_entry_index;
    I64 snapshot_access_tick;
    I64 snapshot_profile_id;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_previous_tick;
    I64 *snapshot_out_previous_tick;
    I64 *snapshot_out_applied_tick;
    I64 *snapshot_out_guard_passed;
    I64 preflight_previous_tick;
    I64 preflight_applied_tick;
    I64 preflight_guard_passed;
    I64 canonical_previous_tick;
    I64 canonical_applied_tick;
    I64 canonical_guard_passed;
    I32 status_preflight;
    I32 status_canonical;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (profile_id != PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        profile_id != PREFIX_CACHE_PROFILE_DEV_LOCAL)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_index = entry_index;
    snapshot_access_tick = access_tick;
    snapshot_profile_id = profile_id;
    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_previous_tick = cache->entries[entry_index].last_used_tick;
    snapshot_out_previous_tick = out_previous_tick;
    snapshot_out_applied_tick = out_applied_tick;
    snapshot_out_guard_passed = out_guard_passed;

    if (snapshot_previous_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status_preflight = PrefixCacheReplayGuardCheckedNoPartialCommitOnly(cache,
                                                                        entry_index,
                                                                        access_tick,
                                                                        profile_id,
                                                                        &preflight_previous_tick,
                                                                        &preflight_applied_tick,
                                                                        &preflight_guard_passed);
    if (status_preflight != PREFIX_CACHE_OK)
        return status_preflight;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_previous_tick < 0 ||
        preflight_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_guard_passed != 0 &&
        preflight_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != preflight_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Re-run canonical guard from the original tick snapshot so diagnostics
    // can compare exact tuples before publishing caller-visible outputs.
    cache->entries[entry_index].last_used_tick = snapshot_previous_tick;

    status_canonical = PrefixCacheReplayGuardChecked(cache,
                                                     entry_index,
                                                     access_tick,
                                                     profile_id,
                                                     &canonical_previous_tick,
                                                     &canonical_applied_tick,
                                                     &canonical_guard_passed);
    if (status_canonical != PREFIX_CACHE_OK)
        return status_canonical;

    if (preflight_previous_tick != canonical_previous_tick ||
        preflight_applied_tick != canonical_applied_tick ||
        preflight_guard_passed != canonical_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != canonical_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_previous_tick = preflight_previous_tick;
    *out_applied_tick = preflight_applied_tick;
    *out_guard_passed = preflight_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                        I64 entry_index,
                                                                        I64 access_tick,
                                                                        I64 profile_id,
                                                                        I64 *out_previous_tick,
                                                                        I64 *out_applied_tick,
                                                                        I64 *out_guard_passed)
{
    I64 snapshot_entry_index;
    I64 snapshot_access_tick;
    I64 snapshot_profile_id;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_previous_tick;
    I64 *snapshot_out_previous_tick;
    I64 *snapshot_out_applied_tick;
    I64 *snapshot_out_guard_passed;
    I64 preflight_previous_tick;
    I64 preflight_applied_tick;
    I64 preflight_guard_passed;
    I64 commit_previous_tick;
    I64 commit_applied_tick;
    I64 commit_guard_passed;
    I32 status_preflight;
    I32 status_commit;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (profile_id != PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        profile_id != PREFIX_CACHE_PROFILE_DEV_LOCAL)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_index = entry_index;
    snapshot_access_tick = access_tick;
    snapshot_profile_id = profile_id;
    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_previous_tick = cache->entries[entry_index].last_used_tick;
    snapshot_out_previous_tick = out_previous_tick;
    snapshot_out_applied_tick = out_applied_tick;
    snapshot_out_guard_passed = out_guard_passed;

    if (snapshot_previous_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status_preflight = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnly(cache,
                                                                                      entry_index,
                                                                                      access_tick,
                                                                                      profile_id,
                                                                                      &preflight_previous_tick,
                                                                                      &preflight_applied_tick,
                                                                                      &preflight_guard_passed);
    if (status_preflight != PREFIX_CACHE_OK)
        return status_preflight;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_previous_tick < 0 ||
        preflight_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_guard_passed != 0 &&
        preflight_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != preflight_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Re-run commit-only guard from the original tick snapshot so parity checks
    // compare exact tuples from the same immutable replay state.
    cache->entries[entry_index].last_used_tick = snapshot_previous_tick;

    status_commit = PrefixCacheReplayGuardCheckedNoPartialCommitOnly(cache,
                                                                     entry_index,
                                                                     access_tick,
                                                                     profile_id,
                                                                     &commit_previous_tick,
                                                                     &commit_applied_tick,
                                                                     &commit_guard_passed);
    if (status_commit != PREFIX_CACHE_OK)
        return status_commit;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_previous_tick < 0 ||
        commit_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_guard_passed != 0 &&
        commit_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != commit_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_previous_tick != commit_previous_tick ||
        preflight_applied_tick != commit_applied_tick ||
        preflight_guard_passed != commit_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_previous_tick = preflight_previous_tick;
    *out_applied_tick = preflight_applied_tick;
    *out_guard_passed = preflight_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,
                                                                                   I64 entry_index,
                                                                                   I64 access_tick,
                                                                                   I64 profile_id,
                                                                                   I64 *out_previous_tick,
                                                                                   I64 *out_applied_tick,
                                                                                   I64 *out_guard_passed)
{
    I64 snapshot_entry_index;
    I64 snapshot_access_tick;
    I64 snapshot_profile_id;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_previous_tick;
    I64 *snapshot_out_previous_tick;
    I64 *snapshot_out_applied_tick;
    I64 *snapshot_out_guard_passed;
    I64 parity_previous_tick;
    I64 parity_applied_tick;
    I64 parity_guard_passed;
    I64 commit_previous_tick;
    I64 commit_applied_tick;
    I64 commit_guard_passed;
    I32 status_parity;
    I32 status_commit;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (profile_id != PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        profile_id != PREFIX_CACHE_PROFILE_DEV_LOCAL)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_index = entry_index;
    snapshot_access_tick = access_tick;
    snapshot_profile_id = profile_id;
    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_previous_tick = cache->entries[entry_index].last_used_tick;
    snapshot_out_previous_tick = out_previous_tick;
    snapshot_out_applied_tick = out_applied_tick;
    snapshot_out_guard_passed = out_guard_passed;

    if (snapshot_previous_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status_parity = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParity(cache,
                                                                                         entry_index,
                                                                                         access_tick,
                                                                                         profile_id,
                                                                                         &parity_previous_tick,
                                                                                         &parity_applied_tick,
                                                                                         &parity_guard_passed);
    if (status_parity != PREFIX_CACHE_OK)
        return status_parity;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_previous_tick < 0 ||
        parity_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_guard_passed != 0 &&
        parity_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != parity_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Re-run commit-only guard from the immutable original tick snapshot so
    // caller-visible outputs are published only after strict tuple parity.
    cache->entries[entry_index].last_used_tick = snapshot_previous_tick;

    status_commit = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnly(cache,
                                                                                   entry_index,
                                                                                   access_tick,
                                                                                   profile_id,
                                                                                   &commit_previous_tick,
                                                                                   &commit_applied_tick,
                                                                                   &commit_guard_passed);
    if (status_commit != PREFIX_CACHE_OK)
        return status_commit;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_previous_tick < 0 ||
        commit_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_guard_passed != 0 &&
        commit_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != commit_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_previous_tick != commit_previous_tick ||
        parity_applied_tick != commit_applied_tick ||
        parity_guard_passed != commit_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_previous_tick = parity_previous_tick;
    *out_applied_tick = parity_applied_tick;
    *out_guard_passed = parity_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                                                I64 entry_index,
                                                                                                I64 access_tick,
                                                                                                I64 profile_id,
                                                                                                I64 *out_previous_tick,
                                                                                                I64 *out_applied_tick,
                                                                                                I64 *out_guard_passed)
{
    I64 snapshot_entry_index;
    I64 snapshot_access_tick;
    I64 snapshot_profile_id;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_previous_tick;
    I64 *snapshot_out_previous_tick;
    I64 *snapshot_out_applied_tick;
    I64 *snapshot_out_guard_passed;
    I64 commit_previous_tick;
    I64 commit_applied_tick;
    I64 commit_guard_passed;
    I64 parity_previous_tick;
    I64 parity_applied_tick;
    I64 parity_guard_passed;
    I32 status_commit;
    I32 status_parity;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (profile_id != PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        profile_id != PREFIX_CACHE_PROFILE_DEV_LOCAL)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_index = entry_index;
    snapshot_access_tick = access_tick;
    snapshot_profile_id = profile_id;
    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_previous_tick = cache->entries[entry_index].last_used_tick;
    snapshot_out_previous_tick = out_previous_tick;
    snapshot_out_applied_tick = out_applied_tick;
    snapshot_out_guard_passed = out_guard_passed;

    if (snapshot_previous_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status_commit = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(cache,
                                                                                                   entry_index,
                                                                                                   access_tick,
                                                                                                   profile_id,
                                                                                                   &commit_previous_tick,
                                                                                                   &commit_applied_tick,
                                                                                                   &commit_guard_passed);
    if (status_commit != PREFIX_CACHE_OK)
        return status_commit;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_previous_tick < 0 ||
        commit_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_guard_passed != 0 &&
        commit_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != commit_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Re-run parity-only diagnostics from immutable snapshot state so this
    // wrapper stays zero-write until strict tuple parity is proven.
    cache->entries[entry_index].last_used_tick = snapshot_previous_tick;

    status_parity = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParity(cache,
                                                                                         entry_index,
                                                                                         access_tick,
                                                                                         profile_id,
                                                                                         &parity_previous_tick,
                                                                                         &parity_applied_tick,
                                                                                         &parity_guard_passed);
    if (status_parity != PREFIX_CACHE_OK)
        return status_parity;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_previous_tick < 0 ||
        parity_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_guard_passed != 0 &&
        parity_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (parity_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != parity_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_previous_tick != parity_previous_tick ||
        commit_applied_tick != parity_applied_tick ||
        commit_guard_passed != parity_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_previous_tick = commit_previous_tick;
    *out_applied_tick = commit_applied_tick;
    *out_guard_passed = commit_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                                                      I64 entry_index,
                                                                                                      I64 access_tick,
                                                                                                      I64 profile_id,
                                                                                                      I64 *out_previous_tick,
                                                                                                      I64 *out_applied_tick,
                                                                                                      I64 *out_guard_passed)
{
    I64 snapshot_entry_index;
    I64 snapshot_access_tick;
    I64 snapshot_profile_id;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_previous_tick;
    I64 *snapshot_out_previous_tick;
    I64 *snapshot_out_applied_tick;
    I64 *snapshot_out_guard_passed;
    I64 preflight_previous_tick;
    I64 preflight_applied_tick;
    I64 preflight_guard_passed;
    I64 commit_previous_tick;
    I64 commit_applied_tick;
    I64 commit_guard_passed;
    I32 status_preflight;
    I32 status_commit;

    if (!cache || !cache->entries ||
        !out_previous_tick || !out_applied_tick || !out_guard_passed)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_previous_tick == out_applied_tick ||
        out_previous_tick == out_guard_passed ||
        out_applied_tick == out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].valid != PREFIX_CACHE_FRESH_VALID)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (profile_id != PREFIX_CACHE_PROFILE_SECURE_LOCAL &&
        profile_id != PREFIX_CACHE_PROFILE_DEV_LOCAL)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_index = entry_index;
    snapshot_access_tick = access_tick;
    snapshot_profile_id = profile_id;
    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_previous_tick = cache->entries[entry_index].last_used_tick;
    snapshot_out_previous_tick = out_previous_tick;
    snapshot_out_applied_tick = out_applied_tick;
    snapshot_out_guard_passed = out_guard_passed;

    if (snapshot_previous_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    status_preflight = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(cache,
                                                                                                                   entry_index,
                                                                                                                   access_tick,
                                                                                                                   profile_id,
                                                                                                                   &preflight_previous_tick,
                                                                                                                   &preflight_applied_tick,
                                                                                                                   &preflight_guard_passed);
    if (status_preflight != PREFIX_CACHE_OK)
        return status_preflight;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_previous_tick < 0 ||
        preflight_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_guard_passed != 0 &&
        preflight_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != preflight_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Re-run commit path from immutable snapshot state so strict tuple parity
    // compares both wrappers from the exact same replay baseline.
    cache->entries[entry_index].last_used_tick = snapshot_previous_tick;

    status_commit = PrefixCacheReplayGuardCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(cache,
                                                                                                   entry_index,
                                                                                                   access_tick,
                                                                                                   profile_id,
                                                                                                   &commit_previous_tick,
                                                                                                   &commit_applied_tick,
                                                                                                   &commit_guard_passed);
    if (status_commit != PREFIX_CACHE_OK)
        return status_commit;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index != snapshot_entry_index ||
        access_tick != snapshot_access_tick ||
        profile_id != snapshot_profile_id)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (out_previous_tick != snapshot_out_previous_tick ||
        out_applied_tick != snapshot_out_applied_tick ||
        out_guard_passed != snapshot_out_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_previous_tick < 0 ||
        commit_applied_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_guard_passed != 0 &&
        commit_guard_passed != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (commit_applied_tick != access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->entries[entry_index].last_used_tick != commit_applied_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (preflight_previous_tick != commit_previous_tick ||
        preflight_applied_tick != commit_applied_tick ||
        preflight_guard_passed != commit_guard_passed)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Atomic publish: write caller outputs once exact tuple parity is proven.
    *out_previous_tick = preflight_previous_tick;
    *out_applied_tick = preflight_applied_tick;
    *out_guard_passed = preflight_guard_passed;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheLookupBestPrefixChecked(PrefixCache *cache,
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
                // Stable tie-break: identical prefix lengths prefer lowest slot.
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
    I64 prefix_tokens;
    I64 staged_best_index;
    I64 staged_best_tokens;
    I64 snapshot_entry_count;
    I64 snapshot_capacity;
    I64 snapshot_query_hash;
    I64 snapshot_max_prompt_tokens;

    if (!cache || !cache->entries || !out_best_index || !out_best_tokens)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_best_index == out_best_tokens)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (query_hash < 0 || max_prompt_tokens < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_entry_count = cache->count;
    snapshot_capacity = cache->capacity;
    snapshot_query_hash = query_hash;
    snapshot_max_prompt_tokens = max_prompt_tokens;

    staged_best_index = -1;
    staged_best_tokens = 0;

    idx = 0;
    while (idx < snapshot_capacity)
    {
        if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID)
        {
            if (cache->entries[idx].prefix_hash < 0 ||
                cache->entries[idx].prefix_tokens < 0 ||
                cache->entries[idx].kv_start_token < 0 ||
                cache->entries[idx].kv_token_count < 0 ||
                cache->entries[idx].last_used_tick < 0)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            if (cache->entries[idx].prefix_hash == snapshot_query_hash)
            {
                prefix_tokens = cache->entries[idx].prefix_tokens;
                if (prefix_tokens <= snapshot_max_prompt_tokens)
                {
                    // Stable tie-break: same prefix length picks lowest slot.
                    if (staged_best_index < 0 ||
                        prefix_tokens > staged_best_tokens ||
                        (prefix_tokens == staged_best_tokens && idx < staged_best_index))
                    {
                        staged_best_index = idx;
                        staged_best_tokens = prefix_tokens;
                    }
                }
            }
        }
        idx++;
    }

    if (staged_best_index < 0)
        return PREFIX_CACHE_ERR_NOT_FOUND;

    // No-partial contract: reject if caller-visible state drifted.
    if (cache->capacity != snapshot_capacity || cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != snapshot_entry_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (query_hash != snapshot_query_hash || query_hash < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (max_prompt_tokens != snapshot_max_prompt_tokens || max_prompt_tokens < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_best_index = staged_best_index;
    *out_best_tokens = staged_best_tokens;
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

I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParityLegacy(PrefixCache *cache,
                                                                                    I64 query_hash,
                                                                                    I64 max_prompt_tokens,
                                                                                    I64 *out_best_index,
                                                                                    I64 *out_best_tokens)
{
    I64 snapshot_entry_count;
    I64 snapshot_capacity;
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
    snapshot_capacity = cache->capacity;
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

    if (cache->capacity != snapshot_capacity)
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

I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParity(PrefixCache *cache,
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

I32 PrefixCacheInvalidateRangeChecked(PrefixCache *cache,
                                      I64 invalidate_start_token,
                                      I64 invalidate_token_count,
                                      I64 *out_removed_count)
{
    I64 idx;
    I64 live_count;
    I64 removed_count;
    I64 invalidate_end_exclusive;
    I64 entry_start_token;
    I64 entry_token_count;
    I64 entry_end_exclusive;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Overflow-safe half-open range [start, end).
    invalidate_end_exclusive = invalidate_start_token + invalidate_token_count;
    if (invalidate_end_exclusive < invalidate_start_token)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    live_count = 0;
    removed_count = 0;

    idx = 0;
    while (idx < cache->capacity)
    {
        if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID)
        {
            entry_start_token = cache->entries[idx].kv_start_token;
            entry_token_count = cache->entries[idx].kv_token_count;
            if (entry_start_token < 0 || entry_token_count < 0)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            entry_end_exclusive = entry_start_token + entry_token_count;
            if (entry_end_exclusive < entry_start_token)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            // Overlap test for half-open windows:
            // [entry_start, entry_end) intersects [invalidate_start, invalidate_end)
            if (entry_token_count > 0 &&
                invalidate_token_count > 0 &&
                invalidate_start_token < entry_end_exclusive &&
                entry_start_token < invalidate_end_exclusive)
            {
                cache->entries[idx].valid = PREFIX_CACHE_FRESH_EMPTY;
                cache->entries[idx].prefix_hash = 0;
                cache->entries[idx].prefix_tokens = 0;
                cache->entries[idx].kv_start_token = 0;
                cache->entries[idx].kv_token_count = 0;
                cache->entries[idx].last_used_tick = 0;
                removed_count++;
            }
            else
            {
                live_count++;
            }
        }
        idx++;
    }

    cache->count = live_count;
    *out_removed_count = removed_count;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnly(PrefixCache *cache,
                                                         I64 invalidate_start_token,
                                                         I64 invalidate_token_count,
                                                         I64 *out_removed_count)
{
    I64 idx;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 invalidate_end_exclusive;
    I64 entry_start_token;
    I64 entry_token_count;
    I64 entry_end_exclusive;
    I64 staged_live_count;
    I64 staged_removed_count;
    I64 canonical_removed_count;
    I32 status;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    invalidate_end_exclusive = invalidate_start_token + invalidate_token_count;
    if (invalidate_end_exclusive < invalidate_start_token)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;

    // Preflight all entry ranges before invoking the mutating path so any
    // parameter/layout error exits with zero state changes.
    staged_live_count = 0;
    staged_removed_count = 0;
    idx = 0;
    while (idx < cache->capacity)
    {
        if (cache->entries[idx].valid == PREFIX_CACHE_FRESH_VALID)
        {
            entry_start_token = cache->entries[idx].kv_start_token;
            entry_token_count = cache->entries[idx].kv_token_count;
            if (entry_start_token < 0 || entry_token_count < 0)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            entry_end_exclusive = entry_start_token + entry_token_count;
            if (entry_end_exclusive < entry_start_token)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            if (entry_token_count > 0 &&
                invalidate_token_count > 0 &&
                invalidate_start_token < entry_end_exclusive &&
                entry_start_token < invalidate_end_exclusive)
                staged_removed_count++;
            else
                staged_live_count++;
        }
        idx++;
    }

    status = PrefixCacheInvalidateRangeChecked(cache,
                                               invalidate_start_token,
                                               invalidate_token_count,
                                               &canonical_removed_count);
    if (status != PREFIX_CACHE_OK)
        return status;

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 ||
        invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count != staged_live_count ||
        canonical_removed_count != staged_removed_count)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count > snapshot_count)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_removed_count = staged_removed_count;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                      I64 invalidate_start_token,
                                                                      I64 invalidate_token_count,
                                                                      I64 *out_removed_count)
{
    I64 idx;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_invalidate_start_token;
    I64 snapshot_invalidate_token_count;
    I64 snapshot_entry_count;
    I64 *snapshot_out_removed_count;
    PrefixCacheEntry *snapshot_entries;
    I64 preflight_removed_count;
    I64 preflight_remaining_count;
    I64 canonical_removed_count;
    I64 canonical_remaining_count;
    I32 status_preflight;
    I32 status_canonical;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_invalidate_start_token = invalidate_start_token;
    snapshot_invalidate_token_count = invalidate_token_count;
    snapshot_entry_count = cache->count;
    snapshot_out_removed_count = out_removed_count;

    snapshot_entries = MAlloc(snapshot_capacity * sizeof(PrefixCacheEntry));
    if (!snapshot_entries)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < snapshot_capacity)
    {
        snapshot_entries[idx] = cache->entries[idx];
        idx++;
    }

    status_preflight = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnly(cache,
                                                                            invalidate_start_token,
                                                                            invalidate_token_count,
                                                                            &preflight_removed_count);
    if (status_preflight != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_preflight;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    preflight_remaining_count = cache->count;
    if (preflight_removed_count < 0 ||
        preflight_remaining_count < 0 ||
        preflight_remaining_count > snapshot_entry_count ||
        preflight_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_removed_count + preflight_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    // Re-run canonical invalidation from the original immutable snapshot so
    // diagnostics can verify exact {removed_count,remaining_count} parity.
    idx = 0;
    while (idx < snapshot_capacity)
    {
        cache->entries[idx] = snapshot_entries[idx];
        idx++;
    }
    cache->count = snapshot_count;

    status_canonical = PrefixCacheInvalidateRangeChecked(cache,
                                                         invalidate_start_token,
                                                         invalidate_token_count,
                                                         &canonical_removed_count);
    if (status_canonical != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_canonical;
    }

    canonical_remaining_count = cache->count;
    if (canonical_removed_count < 0 ||
        canonical_remaining_count < 0 ||
        canonical_remaining_count > snapshot_entry_count ||
        canonical_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (canonical_removed_count + canonical_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_removed_count != canonical_removed_count ||
        preflight_remaining_count != canonical_remaining_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    Free(snapshot_entries);
    *out_removed_count = preflight_removed_count;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                            I64 invalidate_start_token,
                                                                            I64 invalidate_token_count,
                                                                            I64 *out_removed_count)
{
    I64 idx;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_invalidate_start_token;
    I64 snapshot_invalidate_token_count;
    I64 snapshot_entry_count;
    I64 *snapshot_out_removed_count;
    PrefixCacheEntry *snapshot_entries;
    I64 preflight_removed_count;
    I64 preflight_remaining_count;
    I64 commit_removed_count;
    I64 commit_remaining_count;
    I32 status_preflight;
    I32 status_commit;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_invalidate_start_token = invalidate_start_token;
    snapshot_invalidate_token_count = invalidate_token_count;
    snapshot_entry_count = cache->count;
    snapshot_out_removed_count = out_removed_count;

    snapshot_entries = MAlloc(snapshot_capacity * sizeof(PrefixCacheEntry));
    if (!snapshot_entries)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < snapshot_capacity)
    {
        snapshot_entries[idx] = cache->entries[idx];
        idx++;
    }

    status_preflight = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnly(cache,
                                                                                          invalidate_start_token,
                                                                                          invalidate_token_count,
                                                                                          &preflight_removed_count);
    if (status_preflight != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_preflight;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    preflight_remaining_count = cache->count;
    if (preflight_removed_count < 0 ||
        preflight_remaining_count < 0 ||
        preflight_remaining_count > snapshot_entry_count ||
        preflight_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_removed_count + preflight_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    // Re-run commit-only invalidate from the original immutable snapshot so
    // parity checks compare exact tuples before caller-visible publish.
    idx = 0;
    while (idx < snapshot_capacity)
    {
        cache->entries[idx] = snapshot_entries[idx];
        idx++;
    }
    cache->count = snapshot_count;

    status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnly(cache,
                                                                          invalidate_start_token,
                                                                          invalidate_token_count,
                                                                          &commit_removed_count);
    if (status_commit != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_commit;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    commit_remaining_count = cache->count;
    if (commit_removed_count < 0 ||
        commit_remaining_count < 0 ||
        commit_remaining_count > snapshot_entry_count ||
        commit_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (commit_removed_count + commit_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_removed_count != commit_removed_count ||
        preflight_remaining_count != commit_remaining_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    Free(snapshot_entries);
    *out_removed_count = preflight_removed_count;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,
                                                                                      I64 invalidate_start_token,
                                                                                      I64 invalidate_token_count,
                                                                                      I64 *out_removed_count)
{
    I64 idx;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_invalidate_start_token;
    I64 snapshot_invalidate_token_count;
    I64 snapshot_entry_count;
    I64 *snapshot_out_removed_count;
    PrefixCacheEntry *snapshot_entries;
    I64 parity_removed_count;
    I64 parity_remaining_count;
    I64 commit_removed_count;
    I64 commit_remaining_count;
    I32 status_parity;
    I32 status_commit;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_invalidate_start_token = invalidate_start_token;
    snapshot_invalidate_token_count = invalidate_token_count;
    snapshot_entry_count = cache->count;
    snapshot_out_removed_count = out_removed_count;

    snapshot_entries = MAlloc(snapshot_capacity * sizeof(PrefixCacheEntry));
    if (!snapshot_entries)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < snapshot_capacity)
    {
        snapshot_entries[idx] = cache->entries[idx];
        idx++;
    }

    status_parity = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParity(cache,
                                                                                            invalidate_start_token,
                                                                                            invalidate_token_count,
                                                                                            &parity_removed_count);
    if (status_parity != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_parity;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    parity_remaining_count = cache->count;
    if (parity_removed_count < 0 ||
        parity_remaining_count < 0 ||
        parity_remaining_count > snapshot_entry_count ||
        parity_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (parity_removed_count + parity_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    // Re-run preflight-only invalidation from immutable snapshot so commit-only
    // publish is allowed only when full tuple parity holds exactly.
    idx = 0;
    while (idx < snapshot_capacity)
    {
        cache->entries[idx] = snapshot_entries[idx];
        idx++;
    }
    cache->count = snapshot_count;

    status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnly(cache,
                                                                                      invalidate_start_token,
                                                                                      invalidate_token_count,
                                                                                      &commit_removed_count);
    if (status_commit != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_commit;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    commit_remaining_count = cache->count;
    if (commit_removed_count < 0 ||
        commit_remaining_count < 0 ||
        commit_remaining_count > snapshot_entry_count ||
        commit_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (commit_removed_count + commit_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (parity_removed_count != commit_removed_count ||
        parity_remaining_count != commit_remaining_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    Free(snapshot_entries);
    *out_removed_count = parity_removed_count;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                                                    I64 invalidate_start_token,
                                                                                                    I64 invalidate_token_count,
                                                                                                    I64 *out_removed_count)
{
    I64 idx;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_invalidate_start_token;
    I64 snapshot_invalidate_token_count;
    I64 snapshot_entry_count;
    I64 *snapshot_out_removed_count;
    PrefixCacheEntry *snapshot_entries;
    I64 commit_removed_count;
    I64 commit_remaining_count;
    I64 parity_removed_count;
    I64 parity_remaining_count;
    I32 status_commit;
    I32 status_parity;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_invalidate_start_token = invalidate_start_token;
    snapshot_invalidate_token_count = invalidate_token_count;
    snapshot_entry_count = cache->count;
    snapshot_out_removed_count = out_removed_count;

    snapshot_entries = MAlloc(snapshot_capacity * sizeof(PrefixCacheEntry));
    if (!snapshot_entries)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < snapshot_capacity)
    {
        snapshot_entries[idx] = cache->entries[idx];
        idx++;
    }

    status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(cache,
                                                                                                       invalidate_start_token,
                                                                                                       invalidate_token_count,
                                                                                                       &commit_removed_count);
    if (status_commit != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_commit;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    commit_remaining_count = cache->count;
    if (commit_removed_count < 0 ||
        commit_remaining_count < 0 ||
        commit_remaining_count > snapshot_entry_count ||
        commit_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (commit_removed_count + commit_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    // Re-run parity-only diagnostics from immutable snapshot so this wrapper
    // keeps caller output zero-write until strict tuple parity is proven.
    idx = 0;
    while (idx < snapshot_capacity)
    {
        cache->entries[idx] = snapshot_entries[idx];
        idx++;
    }
    cache->count = snapshot_count;

    status_parity = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParity(cache,
                                                                                            invalidate_start_token,
                                                                                            invalidate_token_count,
                                                                                            &parity_removed_count);
    if (status_parity != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_parity;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    parity_remaining_count = cache->count;
    if (parity_removed_count < 0 ||
        parity_remaining_count < 0 ||
        parity_remaining_count > snapshot_entry_count ||
        parity_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (parity_removed_count + parity_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (commit_removed_count != parity_removed_count ||
        commit_remaining_count != parity_remaining_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    Free(snapshot_entries);
    *out_removed_count = commit_removed_count;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                                                          I64 invalidate_start_token,
                                                                                                          I64 invalidate_token_count,
                                                                                                          I64 *out_removed_count)
{
    I64 idx;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_invalidate_start_token;
    I64 snapshot_invalidate_token_count;
    I64 snapshot_entry_count;
    I64 *snapshot_out_removed_count;
    PrefixCacheEntry *snapshot_entries;
    I64 preflight_removed_count;
    I64 preflight_remaining_count;
    I64 commit_removed_count;
    I64 commit_remaining_count;
    I32 status_preflight;
    I32 status_commit;

    if (!cache || !cache->entries || !out_removed_count)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (invalidate_start_token < 0 || invalidate_token_count < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_invalidate_start_token = invalidate_start_token;
    snapshot_invalidate_token_count = invalidate_token_count;
    snapshot_entry_count = cache->count;
    snapshot_out_removed_count = out_removed_count;

    snapshot_entries = MAlloc(snapshot_capacity * sizeof(PrefixCacheEntry));
    if (!snapshot_entries)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    idx = 0;
    while (idx < snapshot_capacity)
    {
        snapshot_entries[idx] = cache->entries[idx];
        idx++;
    }

    status_preflight = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(cache,
                                                                                                                       invalidate_start_token,
                                                                                                                       invalidate_token_count,
                                                                                                                       &preflight_removed_count);
    if (status_preflight != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_preflight;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    preflight_remaining_count = cache->count;
    if (preflight_removed_count < 0 ||
        preflight_remaining_count < 0 ||
        preflight_remaining_count > snapshot_entry_count ||
        preflight_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_removed_count + preflight_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    // Re-run commit wrapper from immutable snapshot so strict diagnostics
    // parity compares both wrappers against the exact same entry layout.
    idx = 0;
    while (idx < snapshot_capacity)
    {
        cache->entries[idx] = snapshot_entries[idx];
        idx++;
    }
    cache->count = snapshot_count;

    status_commit = PrefixCacheInvalidateRangeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(cache,
                                                                                                       invalidate_start_token,
                                                                                                       invalidate_token_count,
                                                                                                       &commit_removed_count);
    if (status_commit != PREFIX_CACHE_OK)
    {
        Free(snapshot_entries);
        return status_commit;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count < 0 ||
        cache->count > cache->capacity)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (invalidate_start_token != snapshot_invalidate_start_token ||
        invalidate_token_count != snapshot_invalidate_token_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (out_removed_count != snapshot_out_removed_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    commit_remaining_count = cache->count;
    if (commit_removed_count < 0 ||
        commit_remaining_count < 0 ||
        commit_remaining_count > snapshot_entry_count ||
        commit_removed_count > snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (commit_removed_count + commit_remaining_count != snapshot_entry_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_removed_count != commit_removed_count ||
        preflight_remaining_count != commit_remaining_count)
    {
        Free(snapshot_entries);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    Free(snapshot_entries);
    *out_removed_count = preflight_removed_count;
    return PREFIX_CACHE_OK;
}

I64 PrefixCacheAuditTupleDigestQ64(I64 entry_index,
                                   I64 prefix_hash,
                                   I64 prefix_tokens,
                                   I64 kv_start_token,
                                   I64 kv_token_count,
                                   I64 last_used_tick)
{
    U64 digest;

    digest = (U64)PREFIX_CACHE_AUDIT_DOMAIN_TAG;
    digest ^= (U64)entry_index;
    digest *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    digest ^= (U64)prefix_hash;
    digest *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    digest ^= (U64)prefix_tokens;
    digest *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    digest ^= (U64)kv_start_token;
    digest *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    digest ^= (U64)kv_token_count;
    digest *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    digest ^= (U64)last_used_tick;
    digest *= (U64)PREFIX_CACHE_HASH_FNV_PRIME;
    return (I64)(digest & (U64)0x7FFFFFFFFFFFFFFF);
}

I32 PrefixCacheExportAuditRowsChecked(PrefixCache *cache,
                                      I64 *row_buffer,
                                      I64 row_capacity,
                                      I64 *out_rows_written)
{
    I64 idx;
    I64 valid_rows;
    I64 write_row;
    I64 row_base;
    PrefixCacheEntry *entry;
    I64 digest_q64;

    if (!cache || !cache->entries || !out_rows_written)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0 || row_capacity < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    // Preflight pass: validate every live tuple and required row count so
    // export is all-or-nothing and Book-of-Truth rows stay deterministic.
    valid_rows = 0;
    idx = 0;
    while (idx < cache->capacity)
    {
        entry = &cache->entries[idx];
        if (entry->valid == PREFIX_CACHE_FRESH_VALID)
        {
            if (entry->prefix_hash < 0 ||
                entry->prefix_tokens < 0 ||
                entry->kv_start_token < 0 ||
                entry->kv_token_count < 0 ||
                entry->last_used_tick < 0)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            if (entry->kv_start_token + entry->kv_token_count < entry->kv_start_token)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            valid_rows++;
        }
        idx++;
    }

    if (valid_rows != cache->count)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (valid_rows > row_capacity)
        return PREFIX_CACHE_ERR_FULL;

    if (valid_rows > 0 && !row_buffer)
        return PREFIX_CACHE_ERR_NULL_PTR;

    write_row = 0;
    idx = 0;
    while (idx < cache->capacity)
    {
        entry = &cache->entries[idx];
        if (entry->valid == PREFIX_CACHE_FRESH_VALID)
        {
            row_base = write_row * PREFIX_CACHE_AUDIT_TUPLE_WIDTH;
            digest_q64 = PrefixCacheAuditTupleDigestQ64(idx,
                                                        entry->prefix_hash,
                                                        entry->prefix_tokens,
                                                        entry->kv_start_token,
                                                        entry->kv_token_count,
                                                        entry->last_used_tick);

            row_buffer[row_base + 0] = idx;
            row_buffer[row_base + 1] = entry->prefix_hash;
            row_buffer[row_base + 2] = entry->prefix_tokens;
            row_buffer[row_base + 3] = entry->kv_start_token;
            row_buffer[row_base + 4] = entry->kv_token_count;
            row_buffer[row_base + 5] = entry->last_used_tick;
            row_buffer[row_base + 6] = digest_q64;
            write_row++;
        }
        idx++;
    }

    *out_rows_written = write_row;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheExportAuditRowsCheckedNoPartial(PrefixCache *cache,
                                               I64 *row_buffer,
                                               I64 row_capacity,
                                               I64 *out_rows_written)
{
    I64 idx;
    I64 valid_rows;
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_row_capacity;
    I64 write_row;
    I64 row_base;
    PrefixCacheEntry *entry;
    I64 digest_q64;

    if (!cache || !cache->entries || !out_rows_written)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0 || row_capacity < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_row_capacity = row_capacity;

    // Preflight all live tuples and capacity before any row write, so
    // diagnostics callers can rely on strict no-partial semantics.
    valid_rows = 0;
    idx = 0;
    while (idx < cache->capacity)
    {
        entry = &cache->entries[idx];
        if (entry->valid == PREFIX_CACHE_FRESH_VALID)
        {
            if (entry->prefix_hash < 0 ||
                entry->prefix_tokens < 0 ||
                entry->kv_start_token < 0 ||
                entry->kv_token_count < 0 ||
                entry->last_used_tick < 0)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            if (entry->kv_start_token + entry->kv_token_count < entry->kv_start_token)
                return PREFIX_CACHE_ERR_BAD_PARAM;

            valid_rows++;
            if (valid_rows < 0)
                return PREFIX_CACHE_ERR_BAD_PARAM;
        }
        idx++;
    }

    if (valid_rows != cache->count)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (valid_rows > row_capacity)
        return PREFIX_CACHE_ERR_FULL;

    if (valid_rows > 0 && !row_buffer)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity != snapshot_capacity ||
        cache->count != snapshot_count ||
        row_capacity != snapshot_row_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    write_row = 0;
    idx = 0;
    while (idx < cache->capacity)
    {
        entry = &cache->entries[idx];
        if (entry->valid == PREFIX_CACHE_FRESH_VALID)
        {
            if (write_row > (0x7FFFFFFFFFFFFFFF / PREFIX_CACHE_AUDIT_TUPLE_WIDTH))
                return PREFIX_CACHE_ERR_BAD_PARAM;

            row_base = write_row * PREFIX_CACHE_AUDIT_TUPLE_WIDTH;
            digest_q64 = PrefixCacheAuditTupleDigestQ64(idx,
                                                        entry->prefix_hash,
                                                        entry->prefix_tokens,
                                                        entry->kv_start_token,
                                                        entry->kv_token_count,
                                                        entry->last_used_tick);

            row_buffer[row_base + 0] = idx;
            row_buffer[row_base + 1] = entry->prefix_hash;
            row_buffer[row_base + 2] = entry->prefix_tokens;
            row_buffer[row_base + 3] = entry->kv_start_token;
            row_buffer[row_base + 4] = entry->kv_token_count;
            row_buffer[row_base + 5] = entry->last_used_tick;
            row_buffer[row_base + 6] = digest_q64;
            write_row++;
        }
        idx++;
    }

    *out_rows_written = write_row;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheExportAuditRowsCheckedNoPartialCommitOnly(PrefixCache *cache,
                                                         I64 *row_buffer,
                                                         I64 row_capacity,
                                                         I64 *out_rows_written)
{
    I64 snapshot_capacity;
    I64 snapshot_count;
    I64 snapshot_row_capacity;
    I64 preflight_rows_written;
    I64 canonical_rows_written;
    I64 alloc_word_count;
    I64 compare_word_count;
    I64 idx;
    I64 *preflight_rows;
    I64 *canonical_rows;
    I32 status_preflight;
    I32 status_canonical;

    if (!cache || !cache->entries || !out_rows_written)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (cache->capacity <= 0 || row_capacity < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_count = cache->count;
    snapshot_row_capacity = row_capacity;

    preflight_rows = NULL;
    canonical_rows = NULL;
    if (row_capacity > 0)
    {
        if (row_capacity > (0x7FFFFFFFFFFFFFFF / PREFIX_CACHE_AUDIT_TUPLE_WIDTH))
            return PREFIX_CACHE_ERR_BAD_PARAM;

        alloc_word_count = row_capacity * PREFIX_CACHE_AUDIT_TUPLE_WIDTH;
        if (alloc_word_count > (0x7FFFFFFFFFFFFFFF / sizeof(I64)))
            return PREFIX_CACHE_ERR_BAD_PARAM;

        preflight_rows = MAlloc(alloc_word_count * sizeof(I64));
        if (!preflight_rows)
            return PREFIX_CACHE_ERR_BAD_PARAM;

        canonical_rows = MAlloc(alloc_word_count * sizeof(I64));
        if (!canonical_rows)
        {
            Free(preflight_rows);
            return PREFIX_CACHE_ERR_BAD_PARAM;
        }
    }

    status_preflight = PrefixCacheExportAuditRowsCheckedNoPartial(cache,
                                                                  preflight_rows,
                                                                  row_capacity,
                                                                  &preflight_rows_written);
    if (status_preflight != PREFIX_CACHE_OK)
    {
        if (canonical_rows)
            Free(canonical_rows);
        if (preflight_rows)
            Free(preflight_rows);
        return status_preflight;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity ||
        row_capacity != snapshot_row_capacity)
    {
        if (canonical_rows)
            Free(canonical_rows);
        if (preflight_rows)
            Free(preflight_rows);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    status_canonical = PrefixCacheExportAuditRowsChecked(cache,
                                                         canonical_rows,
                                                         row_capacity,
                                                         &canonical_rows_written);
    if (status_canonical != PREFIX_CACHE_OK)
    {
        if (canonical_rows)
            Free(canonical_rows);
        if (preflight_rows)
            Free(preflight_rows);
        return status_canonical;
    }

    if (cache->capacity != snapshot_capacity ||
        cache->count != snapshot_count ||
        cache->count < 0 ||
        cache->count > cache->capacity ||
        row_capacity != snapshot_row_capacity)
    {
        if (canonical_rows)
            Free(canonical_rows);
        if (preflight_rows)
            Free(preflight_rows);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_rows_written != canonical_rows_written)
    {
        if (canonical_rows)
            Free(canonical_rows);
        if (preflight_rows)
            Free(preflight_rows);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_rows_written < 0 || preflight_rows_written > row_capacity)
    {
        if (canonical_rows)
            Free(canonical_rows);
        if (preflight_rows)
            Free(preflight_rows);
        return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (preflight_rows_written > 0)
    {
        if (!row_buffer)
        {
            if (canonical_rows)
                Free(canonical_rows);
            if (preflight_rows)
                Free(preflight_rows);
            return PREFIX_CACHE_ERR_NULL_PTR;
        }

        if (preflight_rows_written > (0x7FFFFFFFFFFFFFFF / PREFIX_CACHE_AUDIT_TUPLE_WIDTH))
        {
            if (canonical_rows)
                Free(canonical_rows);
            if (preflight_rows)
                Free(preflight_rows);
            return PREFIX_CACHE_ERR_BAD_PARAM;
        }

        compare_word_count = preflight_rows_written * PREFIX_CACHE_AUDIT_TUPLE_WIDTH;
        idx = 0;
        while (idx < compare_word_count)
        {
            if (preflight_rows[idx] != canonical_rows[idx])
            {
                if (canonical_rows)
                    Free(canonical_rows);
                if (preflight_rows)
                    Free(preflight_rows);
                return PREFIX_CACHE_ERR_BAD_PARAM;
            }
            idx++;
        }

        idx = 0;
        while (idx < compare_word_count)
        {
            row_buffer[idx] = preflight_rows[idx];
            idx++;
        }
    }

    if (canonical_rows)
        Free(canonical_rows);
    if (preflight_rows)
        Free(preflight_rows);

    *out_rows_written = preflight_rows_written;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(PrefixCache *cache,
                                                                                         I64 query_hash,
                                                                                         I64 max_prompt_tokens,
                                                                                         I64 *out_best_index,
                                                                                         I64 *out_best_tokens)
{
    I64 snapshot_entry_count;
    I64 snapshot_max_prompt_tokens;
    I64 snapshot_query_hash;
    I64 parity_best_index;
    I64 parity_best_tokens;
    I64 canonical_best_index;
    I64 canonical_best_tokens;
    I32 status_parity;
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

    if (parity_best_index != canonical_best_index ||
        parity_best_tokens != canonical_best_tokens)
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

    *out_best_index = parity_best_index;
    *out_best_tokens = parity_best_tokens;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnly(PrefixCache *cache,
                                                         I64 prefix_hash,
                                                         I64 prefix_tokens,
                                                         I64 kv_start_token,
                                                         I64 kv_token_count,
                                                         I64 access_tick,
                                                         I64 *out_entry_index,
                                                         I64 *out_inserted_new)
{
    I64 snapshot_prefix_hash;
    I64 snapshot_prefix_tokens;
    I64 snapshot_kv_start_token;
    I64 snapshot_kv_token_count;
    I64 snapshot_count;
    I64 snapshot_capacity;
    I64 snapshot_access_tick;
    I64 entry_index;
    I64 inserted_new;
    I32 status;

    if (!cache || !cache->entries || !out_entry_index || !out_inserted_new)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_entry_index == out_inserted_new)
        return PREFIX_CACHE_ERR_BAD_PARAM;

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

    snapshot_prefix_hash = prefix_hash;
    snapshot_prefix_tokens = prefix_tokens;
    snapshot_kv_start_token = kv_start_token;
    snapshot_kv_token_count = kv_token_count;
    snapshot_count = cache->count;
    snapshot_capacity = cache->capacity;
    snapshot_access_tick = access_tick;

    status = PrefixCacheInsertOrUpdateChecked(cache,
                                              prefix_hash,
                                              prefix_tokens,
                                              kv_start_token,
                                              kv_token_count,
                                              access_tick,
                                              &entry_index,
                                              &inserted_new);
    if (status != PREFIX_CACHE_OK)
        return status;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (prefix_hash != snapshot_prefix_hash ||
        prefix_tokens != snapshot_prefix_tokens ||
        kv_start_token != snapshot_kv_start_token ||
        kv_token_count != snapshot_kv_token_count)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (inserted_new == 0)
    {
        if (cache->count != snapshot_count)
            return PREFIX_CACHE_ERR_BAD_PARAM;
    }
    else
    {
        if (cache->count != snapshot_count &&
            cache->count != snapshot_count + 1)
            return PREFIX_CACHE_ERR_BAD_PARAM;
    }

    if (access_tick != snapshot_access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (entry_index < 0 || entry_index >= cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (inserted_new != 0 && inserted_new != 1)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_entry_index = entry_index;
    *out_inserted_new = inserted_new;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnlyPreflightOnly(PrefixCache *cache,
                                                                      I64 prefix_hash,
                                                                      I64 prefix_tokens,
                                                                      I64 kv_start_token,
                                                                      I64 kv_token_count,
                                                                      I64 access_tick,
                                                                      I64 *out_entry_index,
                                                                      I64 *out_inserted_new)
{
    I64 preflight_entry_index;
    I64 preflight_inserted_new;
    I64 canonical_entry_index;
    I64 canonical_inserted_new;
    I64 snapshot_capacity;
    I64 snapshot_access_tick;
    I32 status_preflight;
    I32 status_canonical;

    if (!cache || !cache->entries || !out_entry_index || !out_inserted_new)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_entry_index == out_inserted_new)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (PrefixCacheValidateTupleChecked(prefix_hash,
                                        prefix_tokens,
                                        kv_start_token,
                                        kv_token_count) != PREFIX_CACHE_OK)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_access_tick = access_tick;

    status_preflight = PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnly(cache,
                                                                            prefix_hash,
                                                                            prefix_tokens,
                                                                            kv_start_token,
                                                                            kv_token_count,
                                                                            access_tick,
                                                                            &preflight_entry_index,
                                                                            &preflight_inserted_new);
    if (status_preflight != PREFIX_CACHE_OK)
        return status_preflight;

    status_canonical = PrefixCacheInsertOrUpdateChecked(cache,
                                                        prefix_hash,
                                                        prefix_tokens,
                                                        kv_start_token,
                                                        kv_token_count,
                                                        access_tick,
                                                        &canonical_entry_index,
                                                        &canonical_inserted_new);
    if (status_canonical != PREFIX_CACHE_OK)
        return status_canonical;

    if (preflight_entry_index != canonical_entry_index ||
        preflight_inserted_new != canonical_inserted_new)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (access_tick != snapshot_access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_entry_index = preflight_entry_index;
    *out_inserted_new = preflight_inserted_new;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                            I64 prefix_hash,
                                                                            I64 prefix_tokens,
                                                                            I64 kv_start_token,
                                                                            I64 kv_token_count,
                                                                            I64 access_tick,
                                                                            I64 *out_entry_index,
                                                                            I64 *out_inserted_new)
{
    I64 parity_entry_index;
    I64 parity_inserted_new;
    I64 canonical_entry_index;
    I64 canonical_inserted_new;
    I64 snapshot_capacity;
    I64 snapshot_access_tick;
    I32 status_parity;
    I32 status_canonical;

    if (!cache || !cache->entries || !out_entry_index || !out_inserted_new)
        return PREFIX_CACHE_ERR_NULL_PTR;

    if (out_entry_index == out_inserted_new)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity <= 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (access_tick < 0)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (PrefixCacheValidateTupleChecked(prefix_hash,
                                        prefix_tokens,
                                        kv_start_token,
                                        kv_token_count) != PREFIX_CACHE_OK)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    snapshot_capacity = cache->capacity;
    snapshot_access_tick = access_tick;

    status_parity = PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnlyPreflightOnly(cache,
                                                                                       prefix_hash,
                                                                                       prefix_tokens,
                                                                                       kv_start_token,
                                                                                       kv_token_count,
                                                                                       access_tick,
                                                                                       &parity_entry_index,
                                                                                       &parity_inserted_new);
    if (status_parity != PREFIX_CACHE_OK)
        return status_parity;

    status_canonical = PrefixCacheInsertOrUpdateCheckedNoPartialCommitOnly(cache,
                                                                            prefix_hash,
                                                                            prefix_tokens,
                                                                            kv_start_token,
                                                                            kv_token_count,
                                                                            access_tick,
                                                                            &canonical_entry_index,
                                                                            &canonical_inserted_new);
    if (status_canonical != PREFIX_CACHE_OK)
        return status_canonical;

    if (parity_entry_index != canonical_entry_index ||
        parity_inserted_new != canonical_inserted_new)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity != snapshot_capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->count < 0 || cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (access_tick != snapshot_access_tick)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    *out_entry_index = parity_entry_index;
    *out_inserted_new = parity_inserted_new;
    return PREFIX_CACHE_OK;
}

I32 PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnlyPreflightOnlyParity(PrefixCache *cache,
                                                                              I64 query_hash,
                                                                              I64 max_prompt_tokens,
                                                                              I64 *out_best_index,
                                                                              I64 *out_best_tokens)
{
    I64 preflight_best_index;
    I64 preflight_best_tokens;
    I64 canonical_best_index;
    I64 canonical_best_tokens;
    I64 snapshot_entry_count;
    I64 snapshot_capacity;
    I64 snapshot_max_prompt_tokens;
    I64 snapshot_query_hash;
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
    snapshot_capacity = cache->capacity;
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

    if (cache->count != snapshot_entry_count ||
        cache->count < 0 ||
        cache->count > cache->capacity)
        return PREFIX_CACHE_ERR_BAD_PARAM;

    if (cache->capacity != snapshot_capacity ||
        cache->capacity <= 0)
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

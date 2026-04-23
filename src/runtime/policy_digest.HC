// policy_digest.HC
// Runtime policy digest emitter for Trinity parity auditing.
//
// Design goals:
// - Integer-only digest path (HolyC/Terry-safe arithmetic).
// - Deterministic digest from explicit policy guard tuple.
// - Transparent bit-level policy snapshot for Book-of-Truth attestations.

#define INFERENCE_POLICY_OK                 0
#define INFERENCE_POLICY_ERR_NULL_PTR       1
#define INFERENCE_POLICY_ERR_BAD_PARAM      2
#define INFERENCE_POLICY_ERR_PROFILE_GUARD  3

#define INFERENCE_POLICY_PROFILE_SECURE_LOCAL 1
#define INFERENCE_POLICY_PROFILE_DEV_LOCAL    2

#define INFERENCE_POLICY_DIGEST_VERSION 1

// 64-bit FNV-1a constants (integer-only, deterministic).
#define INFERENCE_POLICY_FNV_OFFSET_BASIS 1469598103934665603
#define INFERENCE_POLICY_FNV_PRIME        1099511628211

I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default);

// Runtime policy-guard state defaults to strict secure posture.
I64 g_policy_iommu_enabled = 1;
I64 g_policy_bot_dma_log_enabled = 1;
I64 g_policy_bot_mmio_log_enabled = 1;
I64 g_policy_bot_dispatch_log_enabled = 1;
I64 g_policy_quarantine_gate_enabled = 1;
I64 g_policy_hash_manifest_gate_enabled = 1;

Bool InferencePolicyFlagIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

I64 InferencePolicyDigestMixU64(I64 digest_state, I64 value)
{
    // FNV-1a round: xor the 64-bit lane, then multiply by prime.
    // All operations are integer arithmetic.
    digest_state ^= value;
    digest_state *= INFERENCE_POLICY_FNV_PRIME;
    return digest_state;
}

I64 InferencePolicyDigestFinalize(I64 digest_state)
{
    // Integer avalanche to spread low-bit structure across full width.
    // Constants stay in signed 64-bit domain for HolyC compatibility.
    digest_state ^= (digest_state >> 31);
    digest_state *= 1140071481932319845;
    digest_state ^= (digest_state >> 27);
    digest_state *= 7046029254386353131;
    digest_state ^= (digest_state >> 33);
    return digest_state;
}

I32 InferencePolicyRuntimeGuardsSetChecked(I64 iommu_enabled,
                                           I64 bot_dma_log_enabled,
                                           I64 bot_mmio_log_enabled,
                                           I64 bot_dispatch_log_enabled,
                                           I64 quarantine_gate_enabled,
                                           I64 hash_manifest_gate_enabled)
{
    if (!InferencePolicyFlagIsBinary(iommu_enabled) ||
        !InferencePolicyFlagIsBinary(bot_dma_log_enabled) ||
        !InferencePolicyFlagIsBinary(bot_mmio_log_enabled) ||
        !InferencePolicyFlagIsBinary(bot_dispatch_log_enabled) ||
        !InferencePolicyFlagIsBinary(quarantine_gate_enabled) ||
        !InferencePolicyFlagIsBinary(hash_manifest_gate_enabled))
        return INFERENCE_POLICY_ERR_BAD_PARAM;

    g_policy_iommu_enabled = iommu_enabled;
    g_policy_bot_dma_log_enabled = bot_dma_log_enabled;
    g_policy_bot_mmio_log_enabled = bot_mmio_log_enabled;
    g_policy_bot_dispatch_log_enabled = bot_dispatch_log_enabled;
    g_policy_quarantine_gate_enabled = quarantine_gate_enabled;
    g_policy_hash_manifest_gate_enabled = hash_manifest_gate_enabled;

    return INFERENCE_POLICY_OK;
}

I32 InferencePolicyDigestChecked(I64 iommu_enabled,
                                 I64 bot_dma_log_enabled,
                                 I64 bot_mmio_log_enabled,
                                 I64 bot_dispatch_log_enabled,
                                 I64 quarantine_gate_enabled,
                                 I64 hash_manifest_gate_enabled,
                                 I64 *out_policy_digest,
                                 I64 *out_policy_bits,
                                 I64 *out_profile_id)
{
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;
    I64 policy_bits;
    I64 digest_state;
    I32 status;

    if (!out_policy_digest || !out_policy_bits || !out_profile_id)
        return INFERENCE_POLICY_ERR_NULL_PTR;

    *out_policy_digest = 0;
    *out_policy_bits = 0;
    *out_profile_id = 0;

    if (!InferencePolicyFlagIsBinary(iommu_enabled) ||
        !InferencePolicyFlagIsBinary(bot_dma_log_enabled) ||
        !InferencePolicyFlagIsBinary(bot_mmio_log_enabled) ||
        !InferencePolicyFlagIsBinary(bot_dispatch_log_enabled) ||
        !InferencePolicyFlagIsBinary(quarantine_gate_enabled) ||
        !InferencePolicyFlagIsBinary(hash_manifest_gate_enabled))
        return INFERENCE_POLICY_ERR_BAD_PARAM;

    profile_id = 0;
    profile_name = NULL;
    is_secure_default = 0;

    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status != 0)
        return INFERENCE_POLICY_ERR_PROFILE_GUARD;

    if (profile_id != INFERENCE_POLICY_PROFILE_SECURE_LOCAL &&
        profile_id != INFERENCE_POLICY_PROFILE_DEV_LOCAL)
        return INFERENCE_POLICY_ERR_PROFILE_GUARD;

    if (!InferencePolicyFlagIsBinary(is_secure_default))
        return INFERENCE_POLICY_ERR_PROFILE_GUARD;

    // Human-auditable policy bitfield:
    // bit0 iommu, bit1 dma, bit2 mmio, bit3 dispatch,
    // bit4 quarantine, bit5 hash manifest,
    // bit6 secure-default flag, bit7 active profile is secure-local.
    policy_bits = 0;
    policy_bits |= (iommu_enabled << 0);
    policy_bits |= (bot_dma_log_enabled << 1);
    policy_bits |= (bot_mmio_log_enabled << 2);
    policy_bits |= (bot_dispatch_log_enabled << 3);
    policy_bits |= (quarantine_gate_enabled << 4);
    policy_bits |= (hash_manifest_gate_enabled << 5);
    policy_bits |= (is_secure_default << 6);
    policy_bits |= ((profile_id == INFERENCE_POLICY_PROFILE_SECURE_LOCAL) << 7);

    digest_state = INFERENCE_POLICY_FNV_OFFSET_BASIS;

    // Domain-separate runtime policy digest from other hash streams.
    digest_state = InferencePolicyDigestMixU64(digest_state, 0x4951504F4C494359);
    digest_state = InferencePolicyDigestMixU64(digest_state, INFERENCE_POLICY_DIGEST_VERSION);
    digest_state = InferencePolicyDigestMixU64(digest_state, profile_id);
    digest_state = InferencePolicyDigestMixU64(digest_state, is_secure_default);
    digest_state = InferencePolicyDigestMixU64(digest_state, iommu_enabled);
    digest_state = InferencePolicyDigestMixU64(digest_state, bot_dma_log_enabled);
    digest_state = InferencePolicyDigestMixU64(digest_state, bot_mmio_log_enabled);
    digest_state = InferencePolicyDigestMixU64(digest_state, bot_dispatch_log_enabled);
    digest_state = InferencePolicyDigestMixU64(digest_state, quarantine_gate_enabled);
    digest_state = InferencePolicyDigestMixU64(digest_state, hash_manifest_gate_enabled);
    digest_state = InferencePolicyDigestMixU64(digest_state,
                                               INFERENCE_POLICY_PROFILE_SECURE_LOCAL);
    digest_state = InferencePolicyDigestMixU64(digest_state,
                                               INFERENCE_POLICY_PROFILE_DEV_LOCAL);
    digest_state = InferencePolicyDigestMixU64(digest_state, policy_bits);

    *out_policy_digest = InferencePolicyDigestFinalize(digest_state);
    *out_policy_bits = policy_bits;
    *out_profile_id = profile_id;

    return INFERENCE_POLICY_OK;
}

// TempleOS CLI helper: `InferencePolicyDigest;`
// Returns policy digest for current runtime policy guard tuple.
I64 InferencePolicyDigest()
{
    I64 policy_digest;
    I64 policy_bits;
    I64 profile_id;
    I32 status;

    policy_digest = 0;
    policy_bits = 0;
    profile_id = 0;

    status = InferencePolicyDigestChecked(g_policy_iommu_enabled,
                                          g_policy_bot_dma_log_enabled,
                                          g_policy_bot_mmio_log_enabled,
                                          g_policy_bot_dispatch_log_enabled,
                                          g_policy_quarantine_gate_enabled,
                                          g_policy_hash_manifest_gate_enabled,
                                          &policy_digest,
                                          &policy_bits,
                                          &profile_id);
    if (status != INFERENCE_POLICY_OK)
        return -status;

    return policy_digest;
}

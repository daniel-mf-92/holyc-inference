// policy.HC
// GPU dispatch policy gate for inference runtime.
//
// Security contract:
// - secure-local remains the default runtime profile.
// - dev-local is explicit opt-in, but GPU dispatch still requires hard controls.
// - Dispatch is denied unless IOMMU and Book-of-Truth DMA/MMIO/dispatch hooks
//   are all active.

#define GPU_POLICY_OK                 0
#define GPU_POLICY_ERR_NULL_PTR       1
#define GPU_POLICY_ERR_BAD_PARAM      2
#define GPU_POLICY_ERR_PROFILE_GUARD  3
#define GPU_POLICY_ERR_IOMMU_GUARD    4
#define GPU_POLICY_ERR_BOOK_GUARD     5

#define GPU_POLICY_REASON_ALLOW            0
#define GPU_POLICY_REASON_PROFILE_REJECT   1
#define GPU_POLICY_REASON_IOMMU_DISABLED   2
#define GPU_POLICY_REASON_BOOK_HOOKS_MISS  3

#define GPU_POLICY_PROFILE_SECURE_LOCAL 1
#define GPU_POLICY_PROFILE_DEV_LOCAL    2

I32 InferenceProfileStatusChecked(I64 *out_profile_id,
                                  U8 **out_profile_name,
                                  I64 *out_is_secure_default);

Bool GPUPolicyFlagIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

I32 GPUPolicyAllowDispatchChecked(I64 iommu_enabled,
                                  I64 bot_dma_log_enabled,
                                  I64 bot_mmio_log_enabled,
                                  I64 bot_dispatch_log_enabled,
                                  I64 *out_allow_dispatch,
                                  I64 *out_reason_code,
                                  I64 *out_profile_id)
{
    I64 profile_id;
    U8 *profile_name;
    I64 is_secure_default;
    I32 status;

    if (!out_allow_dispatch || !out_reason_code || !out_profile_id)
        return GPU_POLICY_ERR_NULL_PTR;

    *out_allow_dispatch = 0;
    *out_reason_code = GPU_POLICY_REASON_PROFILE_REJECT;
    *out_profile_id = 0;

    if (!GPUPolicyFlagIsBinary(iommu_enabled) ||
        !GPUPolicyFlagIsBinary(bot_dma_log_enabled) ||
        !GPUPolicyFlagIsBinary(bot_mmio_log_enabled) ||
        !GPUPolicyFlagIsBinary(bot_dispatch_log_enabled))
        return GPU_POLICY_ERR_BAD_PARAM;

    profile_id = 0;
    profile_name = NULL;
    is_secure_default = 0;

    status = InferenceProfileStatusChecked(&profile_id,
                                           &profile_name,
                                           &is_secure_default);
    if (status != 0)
        return GPU_POLICY_ERR_PROFILE_GUARD;

    if (profile_id != GPU_POLICY_PROFILE_SECURE_LOCAL &&
        profile_id != GPU_POLICY_PROFILE_DEV_LOCAL)
        return GPU_POLICY_ERR_PROFILE_GUARD;

    *out_profile_id = profile_id;

    if (!iommu_enabled)
    {
        *out_reason_code = GPU_POLICY_REASON_IOMMU_DISABLED;
        return GPU_POLICY_ERR_IOMMU_GUARD;
    }

    // Book-of-Truth guard: DMA/MMIO/dispatch hooks must all be on before
    // device command submission is allowed.
    if (!bot_dma_log_enabled ||
        !bot_mmio_log_enabled ||
        !bot_dispatch_log_enabled)
    {
        *out_reason_code = GPU_POLICY_REASON_BOOK_HOOKS_MISS;
        return GPU_POLICY_ERR_BOOK_GUARD;
    }

    *out_allow_dispatch = 1;
    *out_reason_code = GPU_POLICY_REASON_ALLOW;
    return GPU_POLICY_OK;
}


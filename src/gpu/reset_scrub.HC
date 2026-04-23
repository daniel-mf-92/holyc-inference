// reset_scrub.HC
// GPU reset + memory scrub sequencing helper (WS9-28).
//
// Security contract:
// - Trusted sessions must be fenced by deterministic reset/scrub sequences.
// - Pre-session path: reset first, then scrub GPU partition.
// - Post-session path: scrub first, then reset.
// - Dispatch prerequisites mirror secure gating: IOMMU + Book-of-Truth hooks.
// - Integer-only checked arithmetic; fail closed and avoid partial state publish.

#define GPU_RESET_SCRUB_OK                    0
#define GPU_RESET_SCRUB_ERR_NULL_PTR          1
#define GPU_RESET_SCRUB_ERR_BAD_PARAM         2
#define GPU_RESET_SCRUB_ERR_BAD_STATE         3
#define GPU_RESET_SCRUB_ERR_OVERFLOW          4
#define GPU_RESET_SCRUB_ERR_POLICY_GUARD      5
#define GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD    6

#define GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL  1
#define GPU_RESET_SCRUB_PROFILE_DEV_LOCAL     2

class GPUResetScrubContext
{
    I64 profile_id;
    I64 session_active;

    I64 iommu_enabled;
    I64 bot_dma_log_enabled;
    I64 bot_mmio_log_enabled;
    I64 bot_dispatch_log_enabled;

    I64 total_resets;
    I64 total_scrub_passes;
    I64 total_scrub_blocks;
    I64 total_scrub_bytes;

    // Monotonic sequence for audit transcript ordering.
    I64 sequence_id;
};

Bool GPUResetScrubFlagIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

Bool GPUResetScrubAddChecked(I64 lhs, I64 rhs, I64 *out_sum)
{
    I64 i64_max;

    if (!out_sum)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    i64_max = 0x7FFFFFFFFFFFFFFF;
    if (lhs > (i64_max - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool GPUResetScrubMulChecked(I64 lhs, I64 rhs, I64 *out_product)
{
    I64 i64_max;

    if (!out_product)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (lhs == 0 || rhs == 0)
    {
        *out_product = 0;
        return TRUE;
    }

    i64_max = 0x7FFFFFFFFFFFFFFF;
    if (lhs > (i64_max / rhs))
        return FALSE;

    *out_product = lhs * rhs;
    return TRUE;
}

Bool GPUResetScrubContextPrereqsAreActive(const GPUResetScrubContext *ctx)
{
    if (!ctx)
        return FALSE;

    if (!GPUResetScrubFlagIsBinary(ctx->iommu_enabled) ||
        !GPUResetScrubFlagIsBinary(ctx->bot_dma_log_enabled) ||
        !GPUResetScrubFlagIsBinary(ctx->bot_mmio_log_enabled) ||
        !GPUResetScrubFlagIsBinary(ctx->bot_dispatch_log_enabled))
        return FALSE;

    if (!ctx->iommu_enabled)
        return FALSE;

    if (!ctx->bot_dma_log_enabled ||
        !ctx->bot_mmio_log_enabled ||
        !ctx->bot_dispatch_log_enabled)
        return FALSE;

    return TRUE;
}

I32 GPUResetScrubContextInit(GPUResetScrubContext *ctx,
                             I64 profile_id,
                             I64 iommu_enabled,
                             I64 bot_dma_log_enabled,
                             I64 bot_mmio_log_enabled,
                             I64 bot_dispatch_log_enabled)
{
    if (!ctx)
        return GPU_RESET_SCRUB_ERR_NULL_PTR;

    if (!GPUResetScrubFlagIsBinary(iommu_enabled) ||
        !GPUResetScrubFlagIsBinary(bot_dma_log_enabled) ||
        !GPUResetScrubFlagIsBinary(bot_mmio_log_enabled) ||
        !GPUResetScrubFlagIsBinary(bot_dispatch_log_enabled))
        return GPU_RESET_SCRUB_ERR_BAD_PARAM;

    if (profile_id != GPU_RESET_SCRUB_PROFILE_SECURE_LOCAL &&
        profile_id != GPU_RESET_SCRUB_PROFILE_DEV_LOCAL)
        return GPU_RESET_SCRUB_ERR_BAD_PARAM;

    ctx->profile_id = profile_id;
    ctx->session_active = 0;

    ctx->iommu_enabled = iommu_enabled;
    ctx->bot_dma_log_enabled = bot_dma_log_enabled;
    ctx->bot_mmio_log_enabled = bot_mmio_log_enabled;
    ctx->bot_dispatch_log_enabled = bot_dispatch_log_enabled;

    ctx->total_resets = 0;
    ctx->total_scrub_passes = 0;
    ctx->total_scrub_blocks = 0;
    ctx->total_scrub_bytes = 0;
    ctx->sequence_id = 0;

    return GPU_RESET_SCRUB_OK;
}

I32 GPUResetScrubPlanChecked(I64 partition_nbytes,
                             I64 scrub_chunk_nbytes,
                             I64 scrub_pass_count,
                             I64 *out_blocks_per_pass,
                             I64 *out_total_blocks,
                             I64 *out_total_scrubbed_bytes)
{
    I64 chunk_minus_one;
    I64 ceil_numerator;
    I64 blocks_per_pass;
    I64 total_blocks;
    I64 total_scrubbed_bytes;

    if (!out_blocks_per_pass || !out_total_blocks || !out_total_scrubbed_bytes)
        return GPU_RESET_SCRUB_ERR_NULL_PTR;

    *out_blocks_per_pass = 0;
    *out_total_blocks = 0;
    *out_total_scrubbed_bytes = 0;

    if (partition_nbytes <= 0 || scrub_chunk_nbytes <= 0 || scrub_pass_count <= 0)
        return GPU_RESET_SCRUB_ERR_BAD_PARAM;

    // Integer ceil(partition/chunk): (partition + chunk - 1) / chunk.
    chunk_minus_one = scrub_chunk_nbytes - 1;
    if (!GPUResetScrubAddChecked(partition_nbytes, chunk_minus_one, &ceil_numerator))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    blocks_per_pass = ceil_numerator / scrub_chunk_nbytes;
    if (blocks_per_pass <= 0)
        return GPU_RESET_SCRUB_ERR_BAD_STATE;

    if (!GPUResetScrubMulChecked(blocks_per_pass, scrub_pass_count, &total_blocks))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubMulChecked(partition_nbytes, scrub_pass_count, &total_scrubbed_bytes))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    *out_blocks_per_pass = blocks_per_pass;
    *out_total_blocks = total_blocks;
    *out_total_scrubbed_bytes = total_scrubbed_bytes;

    return GPU_RESET_SCRUB_OK;
}

I32 GPUResetScrubRunPreTrustedSessionChecked(GPUResetScrubContext *ctx,
                                             I64 partition_nbytes,
                                             I64 scrub_chunk_nbytes,
                                             I64 scrub_pass_count,
                                             I64 *out_step_count,
                                             I64 *out_scrub_blocks,
                                             I64 *out_scrubbed_bytes,
                                             I64 *out_sequence_id)
{
    I64 blocks_per_pass;
    I64 total_blocks;
    I64 total_scrubbed_bytes;
    I64 step_count;

    I64 next_total_resets;
    I64 next_total_passes;
    I64 next_total_blocks;
    I64 next_total_bytes;
    I64 next_sequence_id;

    I32 status;

    if (!ctx || !out_step_count || !out_scrub_blocks || !out_scrubbed_bytes || !out_sequence_id)
        return GPU_RESET_SCRUB_ERR_NULL_PTR;

    *out_step_count = 0;
    *out_scrub_blocks = 0;
    *out_scrubbed_bytes = 0;
    *out_sequence_id = 0;

    if (ctx->session_active)
        return GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD;

    if (!GPUResetScrubContextPrereqsAreActive(ctx))
        return GPU_RESET_SCRUB_ERR_POLICY_GUARD;

    status = GPUResetScrubPlanChecked(partition_nbytes,
                                      scrub_chunk_nbytes,
                                      scrub_pass_count,
                                      &blocks_per_pass,
                                      &total_blocks,
                                      &total_scrubbed_bytes);
    if (status != GPU_RESET_SCRUB_OK)
        return status;

    // Pre-session order: 1 reset step + scrub block steps.
    if (!GPUResetScrubAddChecked(1, total_blocks, &step_count))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    // Stage all aggregates before commit to avoid partial state writes.
    if (!GPUResetScrubAddChecked(ctx->total_resets, 1, &next_total_resets))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_scrub_passes, scrub_pass_count, &next_total_passes))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_scrub_blocks, total_blocks, &next_total_blocks))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_scrub_bytes, total_scrubbed_bytes, &next_total_bytes))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->sequence_id, step_count, &next_sequence_id))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    ctx->total_resets = next_total_resets;
    ctx->total_scrub_passes = next_total_passes;
    ctx->total_scrub_blocks = next_total_blocks;
    ctx->total_scrub_bytes = next_total_bytes;
    ctx->sequence_id = next_sequence_id;
    ctx->session_active = 1;

    *out_step_count = step_count;
    *out_scrub_blocks = total_blocks;
    *out_scrubbed_bytes = total_scrubbed_bytes;
    *out_sequence_id = next_sequence_id;

    return GPU_RESET_SCRUB_OK;
}

I32 GPUResetScrubRunPostTrustedSessionChecked(GPUResetScrubContext *ctx,
                                              I64 partition_nbytes,
                                              I64 scrub_chunk_nbytes,
                                              I64 scrub_pass_count,
                                              I64 *out_step_count,
                                              I64 *out_scrub_blocks,
                                              I64 *out_scrubbed_bytes,
                                              I64 *out_sequence_id)
{
    I64 blocks_per_pass;
    I64 total_blocks;
    I64 total_scrubbed_bytes;
    I64 step_count;

    I64 next_total_resets;
    I64 next_total_passes;
    I64 next_total_blocks;
    I64 next_total_bytes;
    I64 next_sequence_id;

    I32 status;

    if (!ctx || !out_step_count || !out_scrub_blocks || !out_scrubbed_bytes || !out_sequence_id)
        return GPU_RESET_SCRUB_ERR_NULL_PTR;

    *out_step_count = 0;
    *out_scrub_blocks = 0;
    *out_scrubbed_bytes = 0;
    *out_sequence_id = 0;

    if (!ctx->session_active)
        return GPU_RESET_SCRUB_ERR_SEQUENCE_GUARD;

    if (!GPUResetScrubContextPrereqsAreActive(ctx))
        return GPU_RESET_SCRUB_ERR_POLICY_GUARD;

    status = GPUResetScrubPlanChecked(partition_nbytes,
                                      scrub_chunk_nbytes,
                                      scrub_pass_count,
                                      &blocks_per_pass,
                                      &total_blocks,
                                      &total_scrubbed_bytes);
    if (status != GPU_RESET_SCRUB_OK)
        return status;

    // Post-session order: scrub block steps + 1 reset step.
    if (!GPUResetScrubAddChecked(total_blocks, 1, &step_count))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_resets, 1, &next_total_resets))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_scrub_passes, scrub_pass_count, &next_total_passes))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_scrub_blocks, total_blocks, &next_total_blocks))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->total_scrub_bytes, total_scrubbed_bytes, &next_total_bytes))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    if (!GPUResetScrubAddChecked(ctx->sequence_id, step_count, &next_sequence_id))
        return GPU_RESET_SCRUB_ERR_OVERFLOW;

    ctx->total_resets = next_total_resets;
    ctx->total_scrub_passes = next_total_passes;
    ctx->total_scrub_blocks = next_total_blocks;
    ctx->total_scrub_bytes = next_total_bytes;
    ctx->sequence_id = next_sequence_id;
    ctx->session_active = 0;

    *out_step_count = step_count;
    *out_scrub_blocks = total_blocks;
    *out_scrubbed_bytes = total_scrubbed_bytes;
    *out_sequence_id = next_sequence_id;

    return GPU_RESET_SCRUB_OK;
}

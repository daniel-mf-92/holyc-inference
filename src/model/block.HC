// block.HC
// Integer-only Transformer block helpers (HolyC runtime path).
//
// IQ-741: residual add kernel with strict no-partial-write semantics.

#define BLOCK_Q16_OK            0
#define BLOCK_Q16_ERR_NULL_PTR  1
#define BLOCK_Q16_ERR_BAD_PARAM 2
#define BLOCK_Q16_ERR_OVERFLOW  4

#define BLOCK_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define BLOCK_I64_MIN_VALUE -0x8000000000000000

I32 FPQ16RMSNormChecked(I64 *input_q16,
                        I64 *gamma_q16,
                        I64 *output_q16,
                        I64 count,
                        I64 eps_q16);

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytes(
    I64 query_row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells,
    I64 *out_last_out_index);

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
    I64 row_count,
    I64 lane_count,
    I64 stage_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

// External checked math/model kernels used by the block apply pipeline.
I32 FPQ16RMSNormChecked(I64 *input_q16,
                        I64 *gamma_q16,
                        I64 *output_q16,
                        I64 count,
                        I64 eps_q16);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 staged_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
    I64 gate_capacity,
    I64 up_capacity,
    I64 out_capacity,
    I64 row_count,
    I64 lane_count,
    I64 stage_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *gate_q16,
    I64 gate_capacity,
    I64 *up_q16,
    I64 up_capacity,
    I64 *out_q16,
    I64 out_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *staging_out_q16,
    I64 staging_out_capacity);

I32 BlockTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > BLOCK_I64_MAX_VALUE - rhs)
        return BLOCK_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < BLOCK_I64_MIN_VALUE - rhs)
        return BLOCK_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return BLOCK_Q16_OK;
}

I32 BlockTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return BLOCK_Q16_OK;
    }

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > (BLOCK_I64_MAX_VALUE / rhs))
                return BLOCK_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (rhs < (BLOCK_I64_MIN_VALUE / lhs))
                return BLOCK_Q16_ERR_OVERFLOW;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < (BLOCK_I64_MIN_VALUE / rhs))
                return BLOCK_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (lhs != 0 && rhs < (BLOCK_I64_MAX_VALUE / lhs))
                return BLOCK_Q16_ERR_OVERFLOW;
        }
    }

    *out_prod = lhs * rhs;
    return BLOCK_Q16_OK;
}

// Residual add kernel (Q16 lanes): out[i] = base[i] + residual[i]
//
// Contracts:
// - capacities/strides/lane_count are measured in I64 lanes
// - all capacities/strides/lane_count are >= 0
// - when lane_count > 0, strides must be >= 1
// - no-partial writes: full overflow/index preflight must pass before writes
I32 TransformerBlockQ16ResidualAddChecked(
    I64 *base_q16,
    I64 base_capacity,
    I64 base_stride_q16,
    I64 *residual_q16,
    I64 residual_capacity,
    I64 residual_stride_q16,
    I64 *out_q16,
    I64 out_capacity,
    I64 out_stride_q16,
    I64 lane_count)
{
    I64 lane_index;
    I64 last_lane;
    I64 required_base_cells;
    I64 required_residual_cells;
    I64 required_out_cells;
    I64 base_index;
    I64 residual_index;
    I64 out_index;
    I64 lane_sum_q16;
    I32 status;

    if (!base_q16 || !residual_q16 || !out_q16)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (base_capacity < 0 || residual_capacity < 0 || out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (base_stride_q16 < 0 || residual_stride_q16 < 0 || out_stride_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return BLOCK_Q16_OK;

    if (base_stride_q16 < 1 || residual_stride_q16 < 1 || out_stride_q16 < 1)
        return BLOCK_Q16_ERR_BAD_PARAM;

    last_lane = lane_count - 1;

    status = BlockTryMulI64Checked(last_lane,
                                   base_stride_q16,
                                   &required_base_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_base_cells,
                                   1,
                                   &required_base_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(last_lane,
                                   residual_stride_q16,
                                   &required_residual_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_residual_cells,
                                   1,
                                   &required_residual_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(last_lane,
                                   out_stride_q16,
                                   &required_out_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_out_cells,
                                   1,
                                   &required_out_cells);
    if (status)
        return status;

    if (required_base_cells > base_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_residual_cells > residual_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    // Preflight overflow for every lane before touching out buffer.
    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = BlockTryMulI64Checked(lane_index,
                                       base_stride_q16,
                                       &base_index);
        if (status)
            return status;
        status = BlockTryMulI64Checked(lane_index,
                                       residual_stride_q16,
                                       &residual_index);
        if (status)
            return status;

        status = BlockTryAddI64Checked(base_q16[base_index],
                                       residual_q16[residual_index],
                                       &lane_sum_q16);
        if (status)
            return status;
    }

    // Commit pass runs only after full preflight succeeds.
    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = BlockTryMulI64Checked(lane_index,
                                       base_stride_q16,
                                       &base_index);
        if (status)
            return status;
        status = BlockTryMulI64Checked(lane_index,
                                       residual_stride_q16,
                                       &residual_index);
        if (status)
            return status;
        status = BlockTryMulI64Checked(lane_index,
                                       out_stride_q16,
                                       &out_index);
        if (status)
            return status;

        status = BlockTryAddI64Checked(base_q16[base_index],
                                       residual_q16[residual_index],
                                       &lane_sum_q16);
        if (status)
            return status;

        out_q16[out_index] = lane_sum_q16;
    }

    return BLOCK_Q16_OK;
}

// Batched residual add (Q16 lanes):
// out[row,lane] = base[row,lane] + residual[row,lane]
//
// Contracts:
// - capacities are in I64 lanes
// - row/lane strides are in lanes
// - no-partial writes: full index + overflow preflight before commit
I32 TransformerBlockQ16ResidualAddRowsCheckedNoPartial(
    I64 *base_q16,
    I64 base_capacity,
    I64 base_row_stride_q16,
    I64 base_lane_stride_q16,
    I64 *residual_q16,
    I64 residual_capacity,
    I64 residual_row_stride_q16,
    I64 residual_lane_stride_q16,
    I64 *out_q16,
    I64 out_capacity,
    I64 out_row_stride_q16,
    I64 out_lane_stride_q16,
    I64 row_count,
    I64 lane_count)
{
    I64 last_row;
    I64 last_lane;
    I64 base_row_max;
    I64 residual_row_max;
    I64 out_row_max;
    I64 base_lane_max;
    I64 residual_lane_max;
    I64 out_lane_max;
    I64 required_base_cells;
    I64 required_residual_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 lane_index;
    I64 base_row_base;
    I64 residual_row_base;
    I64 out_row_base;
    I64 base_lane_offset;
    I64 residual_lane_offset;
    I64 out_lane_offset;
    I64 base_index;
    I64 residual_index;
    I64 out_index;
    I64 lane_sum_q16;
    I32 status;

    if (!base_q16 || !residual_q16 || !out_q16)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (base_capacity < 0 || residual_capacity < 0 || out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (base_row_stride_q16 < 0 || base_lane_stride_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (residual_row_stride_q16 < 0 || residual_lane_stride_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (out_row_stride_q16 < 0 || out_lane_stride_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return BLOCK_Q16_OK;

    if (base_lane_stride_q16 < 1 || residual_lane_stride_q16 < 1 || out_lane_stride_q16 < 1)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (row_count > 1 && (base_row_stride_q16 < 1 ||
                          residual_row_stride_q16 < 1 ||
                          out_row_stride_q16 < 1))
        return BLOCK_Q16_ERR_BAD_PARAM;

    last_row = row_count - 1;
    last_lane = lane_count - 1;

    status = BlockTryMulI64Checked(last_row,
                                   base_row_stride_q16,
                                   &base_row_max);
    if (status)
        return status;
    status = BlockTryMulI64Checked(last_lane,
                                   base_lane_stride_q16,
                                   &base_lane_max);
    if (status)
        return status;
    status = BlockTryAddI64Checked(base_row_max,
                                   base_lane_max,
                                   &required_base_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_base_cells,
                                   1,
                                   &required_base_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(last_row,
                                   residual_row_stride_q16,
                                   &residual_row_max);
    if (status)
        return status;
    status = BlockTryMulI64Checked(last_lane,
                                   residual_lane_stride_q16,
                                   &residual_lane_max);
    if (status)
        return status;
    status = BlockTryAddI64Checked(residual_row_max,
                                   residual_lane_max,
                                   &required_residual_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_residual_cells,
                                   1,
                                   &required_residual_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(last_row,
                                   out_row_stride_q16,
                                   &out_row_max);
    if (status)
        return status;
    status = BlockTryMulI64Checked(last_lane,
                                   out_lane_stride_q16,
                                   &out_lane_max);
    if (status)
        return status;
    status = BlockTryAddI64Checked(out_row_max,
                                   out_lane_max,
                                   &required_out_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_out_cells,
                                   1,
                                   &required_out_cells);
    if (status)
        return status;

    if (required_base_cells > base_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_residual_cells > residual_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    // Full preflight over all rows/lanes before touching out.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       base_row_stride_q16,
                                       &base_row_base);
        if (status)
            return status;
        status = BlockTryMulI64Checked(row_index,
                                       residual_row_stride_q16,
                                       &residual_row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = BlockTryMulI64Checked(lane_index,
                                           base_lane_stride_q16,
                                           &base_lane_offset);
            if (status)
                return status;
            status = BlockTryMulI64Checked(lane_index,
                                           residual_lane_stride_q16,
                                           &residual_lane_offset);
            if (status)
                return status;

            status = BlockTryAddI64Checked(base_row_base,
                                           base_lane_offset,
                                           &base_index);
            if (status)
                return status;
            status = BlockTryAddI64Checked(residual_row_base,
                                           residual_lane_offset,
                                           &residual_index);
            if (status)
                return status;

            status = BlockTryAddI64Checked(base_q16[base_index],
                                           residual_q16[residual_index],
                                           &lane_sum_q16);
            if (status)
                return status;
        }
    }

    // Commit pass runs only after full preflight succeeds.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       base_row_stride_q16,
                                       &base_row_base);
        if (status)
            return status;
        status = BlockTryMulI64Checked(row_index,
                                       residual_row_stride_q16,
                                       &residual_row_base);
        if (status)
            return status;
        status = BlockTryMulI64Checked(row_index,
                                       out_row_stride_q16,
                                       &out_row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = BlockTryMulI64Checked(lane_index,
                                           base_lane_stride_q16,
                                           &base_lane_offset);
            if (status)
                return status;
            status = BlockTryMulI64Checked(lane_index,
                                           residual_lane_stride_q16,
                                           &residual_lane_offset);
            if (status)
                return status;
            status = BlockTryMulI64Checked(lane_index,
                                           out_lane_stride_q16,
                                           &out_lane_offset);
            if (status)
                return status;

            status = BlockTryAddI64Checked(base_row_base,
                                           base_lane_offset,
                                           &base_index);
            if (status)
                return status;
            status = BlockTryAddI64Checked(residual_row_base,
                                           residual_lane_offset,
                                           &residual_index);
            if (status)
                return status;
            status = BlockTryAddI64Checked(out_row_base,
                                           out_lane_offset,
                                           &out_index);
            if (status)
                return status;

            status = BlockTryAddI64Checked(base_q16[base_index],
                                           residual_q16[residual_index],
                                           &lane_sum_q16);
            if (status)
                return status;

            out_q16[out_index] = lane_sum_q16;
        }
    }

    return BLOCK_Q16_OK;
}

// Block-level staging footprint helper for one transformer pass.
//
// The block pipeline stages three dense Q16 slabs of live cells:
// - attention staging
// - FFN staging
// - residual staging
//
// Outputs are deterministic checked diagnostics for caller-managed
// workspace contracts.
I32 TransformerBlockQ16ComputeRequiredBytesChecked(
    I64 row_count,
    I64 lane_count,
    I64 *out_attn_stage_cells,
    I64 *out_ffn_stage_cells,
    I64 *out_residual_cells,
    I64 *out_total_stage_bytes)
{
    I64 attn_stage_cells;
    I64 ffn_stage_cells;
    I64 residual_cells;
    I64 dense_cells;
    I64 total_stage_cells;
    I64 total_stage_bytes;
    I32 status;

    if (!out_attn_stage_cells || !out_ffn_stage_cells ||
        !out_residual_cells || !out_total_stage_bytes)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
    {
        *out_attn_stage_cells = 0;
        *out_ffn_stage_cells = 0;
        *out_residual_cells = 0;
        *out_total_stage_bytes = 0;
        return BLOCK_Q16_OK;
    }

    status = BlockTryMulI64Checked(row_count,
                                   lane_count,
                                   &dense_cells);
    if (status)
        return status;

    // Stage diagnostics locally first so overflow exits never publish
    // partially-updated outputs.
    attn_stage_cells = dense_cells;
    ffn_stage_cells = dense_cells;
    residual_cells = dense_cells;

    status = BlockTryAddI64Checked(dense_cells,
                                   dense_cells,
                                   &total_stage_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(total_stage_cells,
                                   dense_cells,
                                   &total_stage_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(total_stage_cells,
                                   sizeof(I64),
                                   &total_stage_bytes);
    if (status)
        return status;

    *out_attn_stage_cells = attn_stage_cells;
    *out_ffn_stage_cells = ffn_stage_cells;
    *out_residual_cells = residual_cells;
    *out_total_stage_bytes = total_stage_bytes;

    return BLOCK_Q16_OK;
}

// Transformer block preflight-only diagnostics for checked no-partial apply.
//
// Exposes the exact workspace/output contract for one block pass:
// - required_qkv_cells: Q/K/V projection slabs (3 * row_count * lane_count)
// - required_attn_scores: attention score matrix (row_count * row_count)
// - required_ffn_cells: FFN output slab (row_count * lane_count)
// - required_out_cells: final block output slab (row_count * lane_count)
// - required_stage_bytes: bytes required for staging all above diagnostics
//
// This helper never touches model tensors and stages diagnostics locally so
// failures leave caller-owned outputs unchanged.
I32 TransformerBlockQ16ApplyCheckedNoPartialPreflightOnly(
    I64 row_count,
    I64 lane_count,
    I64 *out_required_qkv_cells,
    I64 *out_required_attn_scores,
    I64 *out_required_ffn_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_bytes)
{
    I64 dense_cells;
    I64 required_qkv_cells;
    I64 required_attn_scores;
    I64 required_ffn_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I32 status;

    if (!out_required_qkv_cells || !out_required_attn_scores ||
        !out_required_ffn_cells || !out_required_out_cells ||
        !out_required_stage_bytes)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
    {
        *out_required_qkv_cells = 0;
        *out_required_attn_scores = 0;
        *out_required_ffn_cells = 0;
        *out_required_out_cells = 0;
        *out_required_stage_bytes = 0;
        return BLOCK_Q16_OK;
    }

    status = BlockTryMulI64Checked(row_count,
                                   lane_count,
                                   &dense_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(dense_cells,
                                   3,
                                   &required_qkv_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(row_count,
                                   row_count,
                                   &required_attn_scores);
    if (status)
        return status;

    required_ffn_cells = dense_cells;
    required_out_cells = dense_cells;

    status = BlockTryAddI64Checked(required_qkv_cells,
                                   required_attn_scores,
                                   &required_stage_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_stage_cells,
                                   required_ffn_cells,
                                   &required_stage_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_stage_cells,
                                   required_out_cells,
                                   &required_stage_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(required_stage_cells,
                                   sizeof(I64),
                                   &required_stage_bytes);
    if (status)
        return status;

    *out_required_qkv_cells = required_qkv_cells;
    *out_required_attn_scores = required_attn_scores;
    *out_required_ffn_cells = required_ffn_cells;
    *out_required_out_cells = required_out_cells;
    *out_required_stage_bytes = required_stage_bytes;

    return BLOCK_Q16_OK;
}

// IQ-1044 commit-only checked transformer-block apply.
//
// Pipeline (integer-only):
// 1) RMSNorm per row into caller-provided norm workspace.
// 2) Scaled QK attention scores into caller-provided attention workspace.
// 3) SwiGLU FFN into caller-provided FFN workspace.
// 4) Deterministic attention projection (Q32->Q16) into stage workspace.
// 5) Residual add: out = ffn + projected_attention.
//
// All writes are staged behind full preflight checks on capacities and index
// math so failures leave caller output unchanged.
I32 TransformerBlockQ16ApplyCheckedNoPartialCommitOnly(
    I64 *input_q16,
    I64 input_capacity,
    I64 *norm_gamma_q16,
    I64 norm_gamma_capacity,
    I64 *ffn_up_q16,
    I64 ffn_up_capacity,
    I64 row_count,
    I64 lane_count,
    I64 eps_q16,
    I64 *workspace_norm_q16,
    I64 workspace_norm_capacity,
    I64 *workspace_attn_scores_q32,
    I64 workspace_attn_scores_capacity,
    I64 *workspace_ffn_q16,
    I64 workspace_ffn_capacity,
    I64 *workspace_stage_q16,
    I64 workspace_stage_capacity,
    I64 *out_q16,
    I64 out_capacity)
{
    I64 required_qkv_cells;
    I64 required_attn_scores;
    I64 required_ffn_cells;
    I64 required_out_cells;
    I64 required_stage_bytes;

    I64 dense_cells;
    I64 dense_bytes;

    I64 row_index;
    I64 lane_index;
    I64 row_base;
    I64 attn_row_base;
    I64 lane_token_index;
    I64 attn_index;
    I64 dense_index;

    I64 score_scale_q16;
    I64 projected_q16;
    I32 status;

    if (!input_q16 || !norm_gamma_q16 || !ffn_up_q16)
        return BLOCK_Q16_ERR_NULL_PTR;
    if (!workspace_norm_q16 || !workspace_attn_scores_q32 ||
        !workspace_ffn_q16 || !workspace_stage_q16 || !out_q16)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (input_capacity < 0 || norm_gamma_capacity < 0 || ffn_up_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (workspace_norm_capacity < 0 || workspace_attn_scores_capacity < 0 ||
        workspace_ffn_capacity < 0 || workspace_stage_capacity < 0 ||
        out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0 || eps_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return BLOCK_Q16_OK;

    if (norm_gamma_capacity < lane_count)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = TransformerBlockQ16ApplyCheckedNoPartialPreflightOnly(
        row_count,
        lane_count,
        &required_qkv_cells,
        &required_attn_scores,
        &required_ffn_cells,
        &required_out_cells,
        &required_stage_bytes);
    if (status)
        return status;

    status = BlockTryMulI64Checked(required_out_cells,
                                   sizeof(I64),
                                   &dense_bytes);
    if (status)
        return status;

    if (required_qkv_cells > input_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_qkv_cells > ffn_up_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_out_cells > workspace_norm_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_attn_scores > workspace_attn_scores_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_ffn_cells > workspace_ffn_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_out_cells > workspace_stage_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (required_stage_bytes <= 0 || dense_bytes <= 0)
        return BLOCK_Q16_ERR_OVERFLOW;

    if (workspace_norm_q16 == out_q16)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (workspace_ffn_q16 == out_q16)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (workspace_stage_q16 == out_q16)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (GGUFTensorByteSpansOverlap((U8 *)workspace_norm_q16,
                                   dense_bytes,
                                   (U8 *)workspace_ffn_q16,
                                   dense_bytes))
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (GGUFTensorByteSpansOverlap((U8 *)workspace_norm_q16,
                                   dense_bytes,
                                   (U8 *)workspace_stage_q16,
                                   dense_bytes))
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (GGUFTensorByteSpansOverlap((U8 *)workspace_ffn_q16,
                                   dense_bytes,
                                   (U8 *)workspace_stage_q16,
                                   dense_bytes))
        return BLOCK_Q16_ERR_BAD_PARAM;

    // 1) RMSNorm row-by-row into workspace_norm_q16.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        status = FPQ16RMSNormChecked(input_q16 + row_base,
                                     norm_gamma_q16,
                                     workspace_norm_q16 + row_base,
                                     lane_count,
                                     eps_q16);
        if (status)
            return status;
    }

    // 2) Attention QK score slab (self-attention style: Q = K = normalized rows).
    score_scale_q16 = 1 << 16;
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartial(
        workspace_norm_q16,
        workspace_norm_capacity,
        row_count,
        lane_count,
        workspace_norm_q16,
        workspace_norm_capacity,
        row_count,
        lane_count,
        lane_count,
        score_scale_q16,
        workspace_attn_scores_q32,
        workspace_attn_scores_capacity,
        row_count);
    if (status)
        return status;

    // 3) FFN SwiGLU stage + commit into workspace_ffn_q16.
    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc(
        workspace_norm_q16,
        workspace_norm_capacity,
        ffn_up_q16,
        ffn_up_capacity,
        workspace_ffn_q16,
        workspace_ffn_capacity,
        row_count,
        lane_count,
        workspace_stage_q16,
        workspace_stage_capacity);
    if (status)
        return status;

    // 4) Preflight projection index math over full dense output before writes.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;
        status = BlockTryMulI64Checked(row_index,
                                       row_count,
                                       &attn_row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            lane_token_index = lane_index % row_count;

            status = BlockTryAddI64Checked(attn_row_base,
                                           lane_token_index,
                                           &attn_index);
            if (status)
                return status;
            status = BlockTryAddI64Checked(row_base,
                                           lane_index,
                                           &dense_index);
            if (status)
                return status;

            if (attn_index < 0 || attn_index >= required_attn_scores)
                return BLOCK_Q16_ERR_BAD_PARAM;
            if (dense_index < 0 || dense_index >= required_out_cells)
                return BLOCK_Q16_ERR_BAD_PARAM;
        }
    }

    // Commit projected attention lanes only after full preflight succeeds.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;
        status = BlockTryMulI64Checked(row_index,
                                       row_count,
                                       &attn_row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            lane_token_index = lane_index % row_count;

            status = BlockTryAddI64Checked(attn_row_base,
                                           lane_token_index,
                                           &attn_index);
            if (status)
                return status;
            status = BlockTryAddI64Checked(row_base,
                                           lane_index,
                                           &dense_index);
            if (status)
                return status;

            projected_q16 = workspace_attn_scores_q32[attn_index] >> 16;
            workspace_stage_q16[dense_index] = projected_q16;
        }
    }

    // 5) Residual combine: final out = ffn + projected_attention.
    return TransformerBlockQ16ResidualAddRowsCheckedNoPartial(
        workspace_ffn_q16,
        workspace_ffn_capacity,
        lane_count,
        1,
        workspace_stage_q16,
        workspace_stage_capacity,
        lane_count,
        1,
        out_q16,
        out_capacity,
        lane_count,
        1,
        row_count,
        lane_count);
}

// IQ-1044: checked commit-only transformer block compose helper.
//
// This function composes a full block application pipeline over caller-owned
// staged attention/FFN outputs while preserving strict no-partial publication
// for final `out_q16` writes:
// 1) preflight block workspace/output requirements;
// 2) run attention/FFN required-bytes preflight contracts for shape parity;
// 3) RMSNorm(input) -> staged attention residual add;
// 4) RMSNorm(attn_residual) -> staged FFN residual add;
// 5) commit final output slab only after full checked staging succeeds.
I32 TransformerBlockQ16ApplyCheckedNoPartialCommitOnly(
    I64 *input_q16,
    I64 input_capacity,
    I64 *attn_out_q16,
    I64 attn_out_capacity,
    I64 *ffn_out_q16,
    I64 ffn_out_capacity,
    I64 *gamma_attn_q16,
    I64 gamma_attn_capacity,
    I64 *gamma_ffn_q16,
    I64 gamma_ffn_capacity,
    I64 eps_q16,
    I64 *workspace_q16,
    I64 workspace_capacity,
    I64 *out_q16,
    I64 out_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_required_qkv_cells,
    I64 *out_required_attn_scores,
    I64 *out_required_ffn_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_bytes)
{
    I64 required_qkv_cells;
    I64 required_attn_scores;
    I64 required_ffn_cells;
    I64 required_out_cells;
    I64 required_stage_bytes;
    I64 required_stage_cells;

    I64 attn_required_stage_cells;
    I64 attn_required_stage_bytes;
    I64 attn_required_out_cells;
    I64 attn_last_out_index;

    I64 ffn_required_stage_cells;
    I64 ffn_required_stage_bytes;
    I64 ffn_required_out_cells;

    I64 dense_cells;
    I64 norm0_base;
    I64 residual1_base;
    I64 norm1_base;
    I64 attn_scores_base;
    I64 ffn_stage_base;
    I64 out_stage_base;
    I64 row_index;
    I64 row_base;
    I64 out_index;
    I64 workspace_available;

    I64 snapshot_input_capacity;
    I64 snapshot_attn_out_capacity;
    I64 snapshot_ffn_out_capacity;
    I64 snapshot_gamma_attn_capacity;
    I64 snapshot_gamma_ffn_capacity;
    I64 snapshot_workspace_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I64 *snapshot_input_q16;
    I64 *snapshot_attn_out_q16;
    I64 *snapshot_ffn_out_q16;
    I64 *snapshot_workspace_q16;
    I64 *snapshot_out_q16;
    I32 status;

    if (!input_q16 || !attn_out_q16 || !ffn_out_q16 ||
        !gamma_attn_q16 || !gamma_ffn_q16 ||
        !workspace_q16 || !out_q16 ||
        !out_required_qkv_cells || !out_required_attn_scores ||
        !out_required_ffn_cells || !out_required_out_cells ||
        !out_required_stage_bytes)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (input_capacity < 0 || attn_out_capacity < 0 || ffn_out_capacity < 0 ||
        gamma_attn_capacity < 0 || gamma_ffn_capacity < 0 ||
        workspace_capacity < 0 || out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (eps_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (out_required_qkv_cells == out_required_attn_scores ||
        out_required_qkv_cells == out_required_ffn_cells ||
        out_required_qkv_cells == out_required_out_cells ||
        out_required_qkv_cells == out_required_stage_bytes ||
        out_required_attn_scores == out_required_ffn_cells ||
        out_required_attn_scores == out_required_out_cells ||
        out_required_attn_scores == out_required_stage_bytes ||
        out_required_ffn_cells == out_required_out_cells ||
        out_required_ffn_cells == out_required_stage_bytes ||
        out_required_out_cells == out_required_stage_bytes)
        return BLOCK_Q16_ERR_BAD_PARAM;

    snapshot_input_q16 = input_q16;
    snapshot_attn_out_q16 = attn_out_q16;
    snapshot_ffn_out_q16 = ffn_out_q16;
    snapshot_workspace_q16 = workspace_q16;
    snapshot_out_q16 = out_q16;

    snapshot_input_capacity = input_capacity;
    snapshot_attn_out_capacity = attn_out_capacity;
    snapshot_ffn_out_capacity = ffn_out_capacity;
    snapshot_gamma_attn_capacity = gamma_attn_capacity;
    snapshot_gamma_ffn_capacity = gamma_ffn_capacity;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;

    status = TransformerBlockQ16ApplyCheckedNoPartialPreflightOnly(
        row_count,
        lane_count,
        &required_qkv_cells,
        &required_attn_scores,
        &required_ffn_cells,
        &required_out_cells,
        &required_stage_bytes);
    if (status)
        return status;

    if (!row_count || !lane_count)
    {
        *out_required_qkv_cells = required_qkv_cells;
        *out_required_attn_scores = required_attn_scores;
        *out_required_ffn_cells = required_ffn_cells;
        *out_required_out_cells = required_out_cells;
        *out_required_stage_bytes = required_stage_bytes;
        return BLOCK_Q16_OK;
    }

    status = BlockTryMulI64Checked(row_count,
                                   lane_count,
                                   &dense_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(required_qkv_cells,
                                   sizeof(I64),
                                   &workspace_available);
    if (status)
        return status;

    status = BlockTryMulI64Checked(required_stage_bytes,
                                   1,
                                   &required_stage_cells);
    if (status)
        return status;
    status = BlockTryMulI64Checked(required_stage_cells,
                                   1,
                                   &required_stage_cells);
    if (status)
        return status;
    status = BlockTryMulI64Checked(required_stage_bytes,
                                   1,
                                   &required_stage_bytes);
    if (status)
        return status;

    if (required_stage_bytes % sizeof(I64))
        return BLOCK_Q16_ERR_BAD_PARAM;
    required_stage_cells = required_stage_bytes / sizeof(I64);

    if (input_capacity < required_out_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (attn_out_capacity < required_out_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (ffn_out_capacity < required_out_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (gamma_attn_capacity < lane_count)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (gamma_ffn_capacity < lane_count)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (workspace_capacity < required_stage_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (out_capacity < required_out_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;

    norm0_base = 0;
    residual1_base = dense_cells;
    norm1_base = dense_cells * 2;
    attn_scores_base = required_qkv_cells;

    status = BlockTryAddI64Checked(attn_scores_base,
                                   required_attn_scores,
                                   &ffn_stage_base);
    if (status)
        return status;
    status = BlockTryAddI64Checked(ffn_stage_base,
                                   required_ffn_cells,
                                   &out_stage_base);
    if (status)
        return status;

    status = AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytes(
        row_count,
        row_count,
        workspace_q16 + attn_scores_base,
        workspace_capacity - attn_scores_base,
        workspace_q16 + attn_scores_base,
        workspace_capacity - attn_scores_base,
        &attn_required_stage_cells,
        &attn_required_stage_bytes,
        &attn_required_out_cells,
        &attn_last_out_index);
    if (status)
        return status;

    if (attn_required_stage_cells != required_attn_scores)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (attn_required_out_cells != required_attn_scores)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        row_count,
        lane_count,
        required_ffn_cells,
        &ffn_required_stage_cells,
        &ffn_required_stage_bytes,
        &ffn_required_out_cells);
    if (status)
        return status;

    if (ffn_required_stage_cells != required_ffn_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (ffn_required_out_cells != required_ffn_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        status = FPQ16RMSNormChecked(input_q16 + row_base,
                                     gamma_attn_q16,
                                     workspace_q16 + norm0_base + row_base,
                                     lane_count,
                                     eps_q16);
        if (status)
            return status;
    }

    status = TransformerBlockQ16ResidualAddRowsCheckedNoPartial(
        workspace_q16 + norm0_base,
        dense_cells,
        lane_count,
        1,
        attn_out_q16,
        attn_out_capacity,
        lane_count,
        1,
        workspace_q16 + residual1_base,
        dense_cells,
        lane_count,
        1,
        row_count,
        lane_count);
    if (status)
        return status;

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        status = FPQ16RMSNormChecked(workspace_q16 + residual1_base + row_base,
                                     gamma_ffn_q16,
                                     workspace_q16 + norm1_base + row_base,
                                     lane_count,
                                     eps_q16);
        if (status)
            return status;
    }

    status = TransformerBlockQ16ResidualAddRowsCheckedNoPartial(
        workspace_q16 + norm1_base,
        dense_cells,
        lane_count,
        1,
        ffn_out_q16,
        ffn_out_capacity,
        lane_count,
        1,
        workspace_q16 + out_stage_base,
        workspace_capacity - out_stage_base,
        lane_count,
        1,
        row_count,
        lane_count);
    if (status)
        return status;

    if (snapshot_input_q16 != input_q16 ||
        snapshot_attn_out_q16 != attn_out_q16 ||
        snapshot_ffn_out_q16 != ffn_out_q16 ||
        snapshot_workspace_q16 != workspace_q16 ||
        snapshot_out_q16 != out_q16)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (snapshot_input_capacity != input_capacity ||
        snapshot_attn_out_capacity != attn_out_capacity ||
        snapshot_ffn_out_capacity != ffn_out_capacity ||
        snapshot_gamma_attn_capacity != gamma_attn_capacity ||
        snapshot_gamma_ffn_capacity != gamma_ffn_capacity ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_out_capacity != out_capacity ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return BLOCK_Q16_ERR_BAD_PARAM;

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        for (out_index = 0; out_index < lane_count; out_index++)
        {
            I64 src_index;
            I64 dst_index;

            status = BlockTryAddI64Checked(row_base,
                                           out_index,
                                           &src_index);
            if (status)
                return status;
            status = BlockTryAddI64Checked(row_base,
                                           out_index,
                                           &dst_index);
            if (status)
                return status;

            out_q16[dst_index] = workspace_q16[out_stage_base + src_index];
        }
    }

    *out_required_qkv_cells = required_qkv_cells;
    *out_required_attn_scores = required_attn_scores;
    *out_required_ffn_cells = required_ffn_cells;
    *out_required_out_cells = required_out_cells;
    *out_required_stage_bytes = required_stage_bytes;
    return BLOCK_Q16_OK;
}

I32 BlockQ32ToQ16RoundedChecked(I64 value_q32, I64 *out_q16)
{
    I64 rounding_bias;
    I64 biased_q32;
    I32 status;

    if (!out_q16)
        return BLOCK_Q16_ERR_NULL_PTR;

    rounding_bias = (1 << 15);
    if (value_q32 < 0)
        rounding_bias = -rounding_bias;

    status = BlockTryAddI64Checked(value_q32,
                                   rounding_bias,
                                   &biased_q32);
    if (status)
        return status;

    *out_q16 = biased_q32 >> 16;
    return BLOCK_Q16_OK;
}

// Full block apply with explicit caller-owned workspace contracts.
//
// Pipeline (all integer-only):
// 1) RMSNorm each row into Q16 staging
// 2) Lift Q16->Q32, run checked attention score scaling row-wise
// 3) Round Q32->Q16 attention result
// 4) Run checked FFN SwiGLU rows into Q16 staging
// 5) Residual add input + FFN output into final output
//
// All stages use no-partial kernels; this wrapper performs explicit
// required-bytes preflight for attention/FFN staging before any writes.
I32 TransformerBlockQ16ApplyCheckedNoPartialCommitOnly(
    I64 *input_q16,
    I64 input_capacity,
    I64 *rms_gamma_q16,
    I64 rms_gamma_capacity,
    I64 rms_eps_q16,
    I64 attn_score_scale_q16,
    I64 *ffn_gate_q16,
    I64 ffn_gate_capacity,
    I64 *ffn_up_q16,
    I64 ffn_up_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *workspace_q16,
    I64 workspace_q16_capacity,
    I64 *workspace_q32,
    I64 workspace_q32_capacity,
    I64 *out_q16,
    I64 out_capacity)
{
    I64 dense_cells;
    I64 required_q16_cells;
    I64 required_q32_cells;
    I64 required_q16_bytes;
    I64 required_q32_bytes;
    I64 attn_required_stage_cells;
    I64 attn_required_stage_bytes;
    I64 attn_required_out_cells;
    I64 ffn_required_stage_cells;
    I64 ffn_required_stage_bytes;
    I64 ffn_required_out_cells;
    I64 row_index;
    I64 row_base;
    I64 cell_index;
    I64 *norm_rows_q16;
    I64 *attn_rows_q16;
    I64 *ffn_stage_q16;
    I64 *attn_in_q32;
    I64 *attn_out_q32;
    I64 *attn_stage_q32;
    I32 status;

    if (!input_q16 || !rms_gamma_q16 || !ffn_gate_q16 || !ffn_up_q16 ||
        !workspace_q16 || !workspace_q32 || !out_q16)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (input_capacity < 0 || rms_gamma_capacity < 0 ||
        ffn_gate_capacity < 0 || ffn_up_capacity < 0 ||
        workspace_q16_capacity < 0 || workspace_q32_capacity < 0 ||
        out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (rms_eps_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return BLOCK_Q16_OK;

    // Output must be disjoint from source/staging so no-partial preflight
    // reads observe immutable inputs.
    if (out_q16 == input_q16 || out_q16 == workspace_q16 || out_q16 == workspace_q32)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = BlockTryMulI64Checked(row_count,
                                   lane_count,
                                   &dense_cells);
    if (status)
        return status;

    if (dense_cells > input_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (dense_cells > ffn_gate_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (dense_cells > ffn_up_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (dense_cells > out_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (lane_count > rms_gamma_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = BlockTryAddI64Checked(dense_cells,
                                   dense_cells,
                                   &required_q16_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_q16_cells,
                                   dense_cells,
                                   &required_q16_cells);
    if (status)
        return status;

    status = BlockTryAddI64Checked(dense_cells,
                                   dense_cells,
                                   &required_q32_cells);
    if (status)
        return status;
    status = BlockTryAddI64Checked(required_q32_cells,
                                   dense_cells,
                                   &required_q32_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(required_q16_cells,
                                   sizeof(I64),
                                   &required_q16_bytes);
    if (status)
        return status;
    status = BlockTryMulI64Checked(required_q32_cells,
                                   sizeof(I64),
                                   &required_q32_bytes);
    if (status)
        return status;

    if (required_q16_cells > workspace_q16_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_q32_cells > workspace_q32_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    norm_rows_q16 = workspace_q16;
    attn_rows_q16 = workspace_q16 + dense_cells;
    ffn_stage_q16 = workspace_q16 + (dense_cells * 2);

    attn_in_q32 = workspace_q32;
    attn_out_q32 = workspace_q32 + dense_cells;
    attn_stage_q32 = workspace_q32 + (dense_cells * 2);

    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        attn_in_q32,
        dense_cells,
        row_count,
        lane_count,
        attn_out_q32,
        dense_cells,
        dense_cells,
        &attn_required_stage_cells,
        &attn_required_stage_bytes,
        &attn_required_out_cells);
    if (status)
        return status;
    if (attn_required_out_cells != dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (attn_required_stage_cells > dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        ffn_gate_capacity,
        ffn_up_capacity,
        dense_cells,
        row_count,
        lane_count,
        dense_cells,
        &ffn_required_stage_cells,
        &ffn_required_stage_bytes,
        &ffn_required_out_cells);
    if (status)
        return status;
    if (ffn_required_out_cells != dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (ffn_required_stage_cells > dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;

    // Stage 1: row-wise RMSNorm with shared gamma and epsilon.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        status = FPQ16RMSNormChecked(input_q16 + row_base,
                                     rms_gamma_q16,
                                     norm_rows_q16 + row_base,
                                     lane_count,
                                     rms_eps_q16);
        if (status)
            return status;
    }

    // Stage 2: lift normalized Q16 rows into Q32 so attention score-scale
    // kernel can operate on canonical Q32 score lanes.
    for (cell_index = 0; cell_index < dense_cells; cell_index++)
    {
        status = BlockTryMulI64Checked(norm_rows_q16[cell_index],
                                       (1 << 16),
                                       &attn_in_q32[cell_index]);
        if (status)
            return status;
    }

    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(
        attn_in_q32,
        dense_cells,
        row_count,
        lane_count,
        attn_score_scale_q16,
        attn_out_q32,
        dense_cells,
        attn_stage_q32,
        dense_cells);
    if (status)
        return status;

    // Stage 3: convert scaled Q32 scores back to Q16 activations.
    for (cell_index = 0; cell_index < dense_cells; cell_index++)
    {
        status = BlockQ32ToQ16RoundedChecked(attn_out_q32[cell_index],
                                             &attn_rows_q16[cell_index]);
        if (status)
            return status;
    }

    // Stage 4: checked FFN rows over attention activations.
    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc(
        ffn_gate_q16,
        ffn_gate_capacity,
        ffn_up_q16,
        ffn_up_capacity,
        attn_rows_q16,
        dense_cells,
        row_count,
        lane_count,
        ffn_stage_q16,
        dense_cells);
    if (status)
        return status;

    // Stage 5: residual add back to caller output.
    return TransformerBlockQ16ResidualAddRowsCheckedNoPartial(
        input_q16,
        input_capacity,
        lane_count,
        1,
        attn_rows_q16,
        dense_cells,
        lane_count,
        1,
        out_q16,
        out_capacity,
        lane_count,
        1,
        row_count,
        lane_count);
}

// IQ-1221 checked no-partial forward wrapper with staged residual commit.
//
// Pipeline (integer-only, all staged before final output commit):
// 1) RMSNorm rows into `norm_rows_q16`
// 2) Lift normalized Q16 lanes to Q32 and run checked attention score scale
// 3) Round attention Q32 lanes back to Q16 into `attn_rows_q16`
// 4) Run checked FFN rows into `ffn_rows_q16`
// 5) Compute residual sums into `residual_stage_q16`
// 6) Commit staged residual rows into caller `out_q16` only after full
//    destination index validation succeeds
I32 TransformerBlockQ16ForwardCheckedNoPartial(
    I64 *input_q16,
    I64 input_capacity,
    I64 *rms_gamma_q16,
    I64 rms_gamma_capacity,
    I64 rms_eps_q16,
    I64 attn_score_scale_q16,
    I64 *ffn_gate_q16,
    I64 ffn_gate_capacity,
    I64 *ffn_up_q16,
    I64 ffn_up_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *workspace_q16,
    I64 workspace_q16_capacity,
    I64 *workspace_q32,
    I64 workspace_q32_capacity,
    I64 *out_q16,
    I64 out_capacity)
{
    I64 dense_cells;
    I64 required_q16_cells;
    I64 required_q32_cells;
    I64 required_q16_bytes;
    I64 required_q32_bytes;

    I64 attn_required_stage_cells;
    I64 attn_required_stage_bytes;
    I64 attn_required_out_cells;
    I64 ffn_required_stage_cells;
    I64 ffn_required_stage_bytes;
    I64 ffn_required_out_cells;

    I64 row_index;
    I64 lane_index;
    I64 row_base;
    I64 cell_index;
    I64 out_index;
    I64 stage_index;

    I64 *norm_rows_q16;
    I64 *attn_rows_q16;
    I64 *ffn_rows_q16;
    I64 *residual_stage_q16;
    I64 *attn_in_q32;
    I64 *attn_out_q32;
    I64 *attn_stage_q32;
    I32 status;

    if (!input_q16 || !rms_gamma_q16 || !ffn_gate_q16 || !ffn_up_q16 ||
        !workspace_q16 || !workspace_q32 || !out_q16)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (input_capacity < 0 || rms_gamma_capacity < 0 ||
        ffn_gate_capacity < 0 || ffn_up_capacity < 0 ||
        workspace_q16_capacity < 0 || workspace_q32_capacity < 0 ||
        out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0 || rms_eps_q16 < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return BLOCK_Q16_OK;

    // Output must remain disjoint from input/workspaces so this wrapper can
    // guarantee no partial caller-output writes on any downstream failure.
    if (out_q16 == input_q16 || out_q16 == workspace_q16 || out_q16 == workspace_q32)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = BlockTryMulI64Checked(row_count,
                                   lane_count,
                                   &dense_cells);
    if (status)
        return status;

    if (dense_cells > input_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (dense_cells > ffn_gate_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (dense_cells > ffn_up_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (dense_cells > out_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (lane_count > rms_gamma_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    // Q16 workspace slabs:
    // - norm rows
    // - attention rows (Q16 after rounded projection)
    // - FFN rows
    // - staged residual rows (commit source for out_q16)
    status = BlockTryMulI64Checked(dense_cells,
                                   4,
                                   &required_q16_cells);
    if (status)
        return status;

    // Q32 workspace slabs:
    // - attention input (lifted Q16)
    // - attention output
    // - attention staging scratch
    status = BlockTryMulI64Checked(dense_cells,
                                   3,
                                   &required_q32_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(required_q16_cells,
                                   sizeof(I64),
                                   &required_q16_bytes);
    if (status)
        return status;
    status = BlockTryMulI64Checked(required_q32_cells,
                                   sizeof(I64),
                                   &required_q32_bytes);
    if (status)
        return status;

    if (required_q16_cells > workspace_q16_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (required_q32_cells > workspace_q32_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (required_q16_bytes <= 0 || required_q32_bytes <= 0)
        return BLOCK_Q16_ERR_OVERFLOW;

    if (GGUFTensorByteSpansOverlap((U8 *)workspace_q16,
                                   required_q16_bytes,
                                   (U8 *)out_q16,
                                   dense_cells * sizeof(I64)))
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (GGUFTensorByteSpansOverlap((U8 *)workspace_q32,
                                   required_q32_bytes,
                                   (U8 *)out_q16,
                                   dense_cells * sizeof(I64)))
        return BLOCK_Q16_ERR_BAD_PARAM;

    norm_rows_q16 = workspace_q16;
    attn_rows_q16 = workspace_q16 + dense_cells;
    ffn_rows_q16 = workspace_q16 + (dense_cells * 2);
    residual_stage_q16 = workspace_q16 + (dense_cells * 3);

    attn_in_q32 = workspace_q32;
    attn_out_q32 = workspace_q32 + dense_cells;
    attn_stage_q32 = workspace_q32 + (dense_cells * 2);

    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        attn_in_q32,
        dense_cells,
        row_count,
        lane_count,
        attn_out_q32,
        dense_cells,
        dense_cells,
        &attn_required_stage_cells,
        &attn_required_stage_bytes,
        &attn_required_out_cells);
    if (status)
        return status;
    if (attn_required_out_cells != dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (attn_required_stage_cells > dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        ffn_gate_capacity,
        ffn_up_capacity,
        dense_cells,
        row_count,
        lane_count,
        dense_cells,
        &ffn_required_stage_cells,
        &ffn_required_stage_bytes,
        &ffn_required_out_cells);
    if (status)
        return status;
    if (ffn_required_out_cells != dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (ffn_required_stage_cells > dense_cells)
        return BLOCK_Q16_ERR_BAD_PARAM;

    // Stage 1: row-wise RMSNorm.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        status = FPQ16RMSNormChecked(input_q16 + row_base,
                                     rms_gamma_q16,
                                     norm_rows_q16 + row_base,
                                     lane_count,
                                     rms_eps_q16);
        if (status)
            return status;
    }

    // Stage 2: Q16->Q32 lift before attention score scaling.
    for (cell_index = 0; cell_index < dense_cells; cell_index++)
    {
        status = BlockTryMulI64Checked(norm_rows_q16[cell_index],
                                       (1 << 16),
                                       &attn_in_q32[cell_index]);
        if (status)
            return status;
    }

    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(
        attn_in_q32,
        dense_cells,
        row_count,
        lane_count,
        attn_score_scale_q16,
        attn_out_q32,
        dense_cells,
        attn_stage_q32,
        dense_cells);
    if (status)
        return status;

    // Stage 3: project Q32 attention lanes back into Q16 lanes.
    for (cell_index = 0; cell_index < dense_cells; cell_index++)
    {
        status = BlockQ32ToQ16RoundedChecked(attn_out_q32[cell_index],
                                             &attn_rows_q16[cell_index]);
        if (status)
            return status;
    }

    // Stage 4: FFN on staged lanes.
    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc(
        ffn_gate_q16,
        ffn_gate_capacity,
        ffn_up_q16,
        ffn_up_capacity,
        ffn_rows_q16,
        dense_cells,
        row_count,
        lane_count,
        attn_rows_q16,
        dense_cells);
    if (status)
        return status;

    // Stage 5: residual add into staging area, never directly into caller out.
    status = TransformerBlockQ16ResidualAddRowsCheckedNoPartial(
        input_q16,
        input_capacity,
        lane_count,
        1,
        ffn_rows_q16,
        dense_cells,
        lane_count,
        1,
        residual_stage_q16,
        dense_cells,
        lane_count,
        1,
        row_count,
        lane_count);
    if (status)
        return status;

    // Stage 6a: validate every destination index before any output write.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = BlockTryAddI64Checked(row_base,
                                           lane_index,
                                           &out_index);
            if (status)
                return status;
            if (out_index < 0 || out_index >= dense_cells)
                return BLOCK_Q16_ERR_BAD_PARAM;
            if (out_index >= out_capacity)
                return BLOCK_Q16_ERR_BAD_PARAM;

            status = BlockTryAddI64Checked(row_base,
                                           lane_index,
                                           &stage_index);
            if (status)
                return status;
            if (stage_index < 0 || stage_index >= dense_cells)
                return BLOCK_Q16_ERR_BAD_PARAM;
        }
    }

    // Stage 6b: commit staged residual rows to caller output.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = BlockTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = BlockTryAddI64Checked(row_base,
                                           lane_index,
                                           &out_index);
            if (status)
                return status;

            status = BlockTryAddI64Checked(row_base,
                                           lane_index,
                                           &stage_index);
            if (status)
                return status;

            out_q16[out_index] = residual_stage_q16[stage_index];
        }
    }

    return BLOCK_Q16_OK;
}

// IQ-1050 diagnostics-only preflight companion.
//
// Purpose:
// - derive commit-only workspace contracts for the integer-only block pipeline;
// - enforce immutable snapshot + tuple parity before publishing outputs;
// - perform zero writes to model tensors.
//
// Published tuple (cells/bytes):
// - dense_cells            = row_count * lane_count
// - required_q16_cells     = dense_cells * 3 (norm, attn_rows, ffn_stage)
// - required_q32_cells     = dense_cells * 3 (attn_in, attn_out, attn_stage)
// - required_q16_bytes     = required_q16_cells * sizeof(I64)
// - required_q32_bytes     = required_q32_cells * sizeof(I64)
I32 TransformerBlockQ16ApplyCheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyPreflightOnly_1050(
    I64 row_count,
    I64 lane_count,
    I64 input_capacity,
    I64 rms_gamma_capacity,
    I64 ffn_gate_capacity,
    I64 ffn_up_capacity,
    I64 workspace_q16_capacity,
    I64 workspace_q32_capacity,
    I64 out_capacity,
    I64 *out_dense_cells,
    I64 *out_required_q16_cells,
    I64 *out_required_q32_cells,
    I64 *out_required_q16_bytes,
    I64 *out_required_q32_bytes)
{
    I64 staged_dense_cells;
    I64 staged_required_q16_cells;
    I64 staged_required_q32_cells;
    I64 staged_required_q16_bytes;
    I64 staged_required_q32_bytes;

    I64 recomputed_dense_cells;
    I64 recomputed_required_q16_cells;
    I64 recomputed_required_q32_cells;
    I64 recomputed_required_q16_bytes;
    I64 recomputed_required_q32_bytes;

    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I64 snapshot_input_capacity;
    I64 snapshot_rms_gamma_capacity;
    I64 snapshot_ffn_gate_capacity;
    I64 snapshot_ffn_up_capacity;
    I64 snapshot_workspace_q16_capacity;
    I64 snapshot_workspace_q32_capacity;
    I64 snapshot_out_capacity;
    I32 status;

    if (!out_dense_cells || !out_required_q16_cells || !out_required_q32_cells ||
        !out_required_q16_bytes || !out_required_q32_bytes)
        return BLOCK_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_q16_cells ||
        out_dense_cells == out_required_q32_cells ||
        out_dense_cells == out_required_q16_bytes ||
        out_dense_cells == out_required_q32_bytes ||
        out_required_q16_cells == out_required_q32_cells ||
        out_required_q16_cells == out_required_q16_bytes ||
        out_required_q16_cells == out_required_q32_bytes ||
        out_required_q32_cells == out_required_q16_bytes ||
        out_required_q32_cells == out_required_q32_bytes ||
        out_required_q16_bytes == out_required_q32_bytes)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || lane_count < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (input_capacity < 0 || rms_gamma_capacity < 0 ||
        ffn_gate_capacity < 0 || ffn_up_capacity < 0 ||
        workspace_q16_capacity < 0 || workspace_q32_capacity < 0 ||
        out_capacity < 0)
        return BLOCK_Q16_ERR_BAD_PARAM;

    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_input_capacity = input_capacity;
    snapshot_rms_gamma_capacity = rms_gamma_capacity;
    snapshot_ffn_gate_capacity = ffn_gate_capacity;
    snapshot_ffn_up_capacity = ffn_up_capacity;
    snapshot_workspace_q16_capacity = workspace_q16_capacity;
    snapshot_workspace_q32_capacity = workspace_q32_capacity;
    snapshot_out_capacity = out_capacity;

    // Canonical staging pass.
    status = BlockTryMulI64Checked(row_count,
                                   lane_count,
                                   &staged_dense_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(staged_dense_cells,
                                   3,
                                   &staged_required_q16_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(staged_dense_cells,
                                   3,
                                   &staged_required_q32_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(staged_required_q16_cells,
                                   sizeof(I64),
                                   &staged_required_q16_bytes);
    if (status)
        return status;

    status = BlockTryMulI64Checked(staged_required_q32_cells,
                                   sizeof(I64),
                                   &staged_required_q32_bytes);
    if (status)
        return status;

    if (staged_dense_cells)
    {
        if (staged_dense_cells > input_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
        if (staged_dense_cells > ffn_gate_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
        if (staged_dense_cells > ffn_up_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
        if (staged_dense_cells > out_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
        if (lane_count > rms_gamma_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
        if (staged_required_q16_cells > workspace_q16_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
        if (staged_required_q32_cells > workspace_q32_capacity)
            return BLOCK_Q16_ERR_BAD_PARAM;
    }

    // Recompute from immutable snapshots and demand strict tuple parity.
    status = BlockTryMulI64Checked(snapshot_row_count,
                                   snapshot_lane_count,
                                   &recomputed_dense_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(recomputed_dense_cells,
                                   3,
                                   &recomputed_required_q16_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(recomputed_dense_cells,
                                   3,
                                   &recomputed_required_q32_cells);
    if (status)
        return status;

    status = BlockTryMulI64Checked(recomputed_required_q16_cells,
                                   sizeof(I64),
                                   &recomputed_required_q16_bytes);
    if (status)
        return status;

    status = BlockTryMulI64Checked(recomputed_required_q32_cells,
                                   sizeof(I64),
                                   &recomputed_required_q32_bytes);
    if (status)
        return status;

    if (snapshot_row_count != row_count || snapshot_lane_count != lane_count)
        return BLOCK_Q16_ERR_BAD_PARAM;
    if (snapshot_input_capacity != input_capacity ||
        snapshot_rms_gamma_capacity != rms_gamma_capacity ||
        snapshot_ffn_gate_capacity != ffn_gate_capacity ||
        snapshot_ffn_up_capacity != ffn_up_capacity ||
        snapshot_workspace_q16_capacity != workspace_q16_capacity ||
        snapshot_workspace_q32_capacity != workspace_q32_capacity ||
        snapshot_out_capacity != out_capacity)
        return BLOCK_Q16_ERR_BAD_PARAM;

    if (staged_dense_cells != recomputed_dense_cells ||
        staged_required_q16_cells != recomputed_required_q16_cells ||
        staged_required_q32_cells != recomputed_required_q32_cells ||
        staged_required_q16_bytes != recomputed_required_q16_bytes ||
        staged_required_q32_bytes != recomputed_required_q32_bytes)
        return BLOCK_Q16_ERR_BAD_PARAM;

    *out_dense_cells = staged_dense_cells;
    *out_required_q16_cells = staged_required_q16_cells;
    *out_required_q32_cells = staged_required_q32_cells;
    *out_required_q16_bytes = staged_required_q16_bytes;
    *out_required_q32_bytes = staged_required_q32_bytes;
    return BLOCK_Q16_OK;
}

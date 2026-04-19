// attention.HC
// Integer-only Q16 attention score helpers for HolyC inference.
//
// This file computes per-token Q·K dot rows for one query head row:
//   score_q32[t] = Sum_i ( q_row_q16[i] * k_rows_q16[t*stride + i] )
//
// Inputs are Q16.16 fixed-point values in I64 lanes.
// The product of two Q16 lanes is Q32.32 (stored in I64 integer domain).
// We keep the full Q32 accumulator (no right shift) so softmax/logit stages
// can decide where and how to rescale.

#define ATTN_Q16_OK            0
#define ATTN_Q16_ERR_NULL_PTR  1
#define ATTN_Q16_ERR_BAD_PARAM 2
#define ATTN_Q16_ERR_OVERFLOW  3

#define ATTN_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define ATTN_I64_MIN_VALUE -0x8000000000000000
#define ATTN_U64_MAX_VALUE  0xFFFFFFFFFFFFFFFF
#define ATTN_Q16_SHIFT       16

I32 AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 in_score_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_row_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells)
{
    I64 in_row_cells;
    I64 out_row_cells;
    I64 last_row_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_last_row_base_index ||
        !out_required_in_cells || !out_required_out_cells ||
        !out_required_stage_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 0 || out_score_stride < 0 || row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
    {
        *out_last_row_base_index = 0;
        *out_required_in_cells = 0;
        *out_required_out_cells = 0;
        *out_required_stage_cells = 0;
        return ATTN_Q16_OK;
    }

    if (in_score_stride < 1 || out_score_stride < 1 || row_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(out_row_cells,
                                       1,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (row_stride < in_row_cells || row_stride < out_row_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       row_stride,
                                       &last_row_base_index);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(last_row_base_index,
                                       in_row_cells,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryAddI64Checked(last_row_base_index,
                                       out_row_cells,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    *out_last_row_base_index = last_row_base_index;
    *out_required_in_cells = required_in_cells;
    *out_required_out_cells = required_out_cells;
    *out_required_stage_cells = required_stage_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsChecked(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_row_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells);

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

I32 AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

U64 AttentionAbsToU64(I64 x)
{
    if (x >= 0)
        return (U64)x;

    // Keep I64_MIN representable using two's-complement identity.
    return ((U64)(-(x + 1))) + 1;
}

I32 AttentionApplySignFromMagnitudeChecked(U64 mag, Bool is_negative, I64 *out_value)
{
    if (!out_value)
        return ATTN_Q16_ERR_NULL_PTR;

    if (is_negative)
    {
        if (mag > 0x8000000000000000)
            return ATTN_Q16_ERR_OVERFLOW;

        if (mag == 0x8000000000000000)
        {
            *out_value = ATTN_I64_MIN_VALUE;
            return ATTN_Q16_OK;
        }

        *out_value = -(I64)mag;
        return ATTN_Q16_OK;
    }

    if (mag > (U64)ATTN_I64_MAX_VALUE)
        return ATTN_Q16_ERR_OVERFLOW;

    *out_value = (I64)mag;
    return ATTN_Q16_OK;
}

I32 AttentionTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return ATTN_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > ATTN_I64_MAX_VALUE - rhs)
        return ATTN_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < ATTN_I64_MIN_VALUE - rhs)
        return ATTN_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return ATTN_Q16_OK;
}

I32 AttentionTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    U64 abs_lhs;
    U64 abs_rhs;
    U64 abs_prod;
    Bool is_negative;

    if (!out_prod)
        return ATTN_Q16_ERR_NULL_PTR;

    abs_lhs = AttentionAbsToU64(lhs);
    abs_rhs = AttentionAbsToU64(rhs);
    is_negative = ((lhs < 0) ^ (rhs < 0));

    if (abs_lhs && abs_rhs && abs_lhs > (ATTN_U64_MAX_VALUE / abs_rhs))
        return ATTN_Q16_ERR_OVERFLOW;

    abs_prod = abs_lhs * abs_rhs;
    return AttentionApplySignFromMagnitudeChecked(abs_prod,
                                                  is_negative,
                                                  out_prod);
}

I32 AttentionQ32ScaleByQ16Checked(I64 value_q32,
                                  I64 scale_q16,
                                  I64 *out_value_q32)
{
    U64 abs_value;
    U64 abs_scale;
    U64 abs_prod;
    U64 round_bias;
    U64 rounded;
    Bool is_negative;

    if (!out_value_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    abs_value = AttentionAbsToU64(value_q32);
    abs_scale = AttentionAbsToU64(scale_q16);
    is_negative = ((value_q32 < 0) ^ (scale_q16 < 0));

    if (abs_value && abs_scale && abs_value > (ATTN_U64_MAX_VALUE / abs_scale))
        return ATTN_Q16_ERR_OVERFLOW;

    abs_prod = abs_value * abs_scale;
    round_bias = ((U64)1) << (ATTN_Q16_SHIFT - 1);

    if (abs_prod > ATTN_U64_MAX_VALUE - round_bias)
        rounded = ATTN_U64_MAX_VALUE >> ATTN_Q16_SHIFT;
    else
        rounded = (abs_prod + round_bias) >> ATTN_Q16_SHIFT;

    return AttentionApplySignFromMagnitudeChecked(rounded,
                                                  is_negative,
                                                  out_value_q32);
}

I32 AttentionQ16ComputeQKDotRowChecked(I64 *q_row_q16,
                                       I64 q_row_capacity,
                                       I64 *k_rows_q16,
                                       I64 k_rows_capacity,
                                       I64 token_count,
                                       I64 k_row_stride_q16,
                                       I64 head_dim,
                                       I64 *out_scores_q32,
                                       I64 out_scores_capacity,
                                       I64 out_score_stride)
{
    I64 required_k_cells;
    I64 required_out_cells;
    I64 token_index;
    I64 token_base;
    I64 out_base;
    I64 lane_index;
    I64 product_q32;
    I64 score_q32;
    I32 status;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || k_row_stride_q16 < 0 || head_dim < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Query row must expose the active head dimension.
    if (head_dim > q_row_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // If we have tokens, each K row stride must cover the active head lanes,
    // and output stride must reserve at least one cell per token score.
    if (token_count > 0 && k_row_stride_q16 < head_dim)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count > 0 && out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Full K-matrix span check: token_count rows spaced by stride.
    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_k_cells > k_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Output span check for deterministic row-major score layout:
    // score[t] stored at out_scores_q32[t * out_score_stride].
    if (token_count == 0)
        return ATTN_Q16_OK;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(required_out_cells,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           k_row_stride_q16,
                                           &token_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
            return status;

        score_q32 = 0;
        for (lane_index = 0; lane_index < head_dim; lane_index++)
        {
            status = AttentionTryMulI64Checked(q_row_q16[lane_index],
                                               k_rows_q16[token_base + lane_index],
                                               &product_q32);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(score_q32,
                                               product_q32,
                                               &score_q32);
            if (status != ATTN_Q16_OK)
                return status;
        }

        out_scores_q32[out_base] = score_q32;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleChecked(I64 *in_scores_q32,
                                       I64 in_scores_capacity,
                                       I64 token_count,
                                       I64 in_score_stride,
                                       I64 score_scale_q16,
                                       I64 *out_scores_q32,
                                       I64 out_scores_capacity,
                                       I64 out_score_stride)
{
    I64 required_in_cells;
    I64 required_out_cells;
    I64 token_index;
    I64 in_base;
    I64 out_base;
    I64 scaled_score_q32;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || in_score_stride < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
        return ATTN_Q16_OK;

    // Strides must expose one scalar score per token.
    if (in_score_stride < 1 || out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Validate full source span: token_count rows spaced by in_score_stride.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(required_in_cells,
                                       1,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Validate full destination span: token_count rows spaced by out_score_stride.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(required_out_cells,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           in_score_stride,
                                           &in_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionQ32ScaleByQ16Checked(in_scores_q32[in_base],
                                               score_scale_q16,
                                               &scaled_score_q32);
        if (status != ATTN_Q16_OK)
            return status;

        out_scores_q32[out_base] = scaled_score_q32;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleCheckedNoPartial(I64 *in_scores_q32,
                                                I64 in_scores_capacity,
                                                I64 token_count,
                                                I64 in_score_stride,
                                                I64 score_scale_q16,
                                                I64 *out_scores_q32,
                                                I64 out_scores_capacity,
                                                I64 out_score_stride)
{
    I64 last_in_base_index;
    I64 last_out_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 token_index;
    I64 out_base;
    I64 *staged_scores_q32;
    I32 status;

    status = AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        in_score_stride,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        &last_in_base_index,
        &last_out_base_index,
        &required_in_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!token_count)
        return ATTN_Q16_OK;

    // Alias-safe no-partial staging: compute all scaled scores in an isolated
    // dense buffer first, then commit to caller output only on full success.
    stage_cells = token_count;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ApplyScoreScaleChecked(in_scores_q32,
                                                in_scores_capacity,
                                                token_count,
                                                in_score_stride,
                                                score_scale_q16,
                                                staged_scores_q32,
                                                stage_cells,
                                                1);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        out_scores_q32[out_base] = staged_scores_q32[token_index];
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 in_score_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells)
{
    I64 total_in_cells;
    I64 total_out_cells;
    I64 last_in_base_index;
    I64 last_out_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_last_in_base_index ||
        !out_last_out_base_index || !out_required_in_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || in_score_stride < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
    {
        *out_last_in_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_in_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    // One score lane per token for both source and destination paths.
    if (in_score_stride < 1 || out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Explicit full-span multiply gates keep stride math auditable before
    // calling the checked scaler core.
    status = AttentionTryMulI64Checked(token_count,
                                       in_score_stride,
                                       &total_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count,
                                       out_score_stride,
                                       &total_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &last_in_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_in_base_index,
                                       1,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &last_out_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_out_base_index,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_last_in_base_index = last_in_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_in_cells = required_in_cells;
    *out_required_out_cells = required_out_cells;

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleCheckedDefaultStride(I64 *in_scores_q32,
                                                    I64 in_scores_capacity,
                                                    I64 token_count,
                                                    I64 score_scale_q16,
                                                    I64 *out_scores_q32,
                                                    I64 out_scores_capacity)
{
    I64 default_score_stride;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-548 default-stride contract for row-major attention score buffers:
    // score[token_index] is addressed at base + token_index * token_count.
    // The checked core owns all overflow and span validation for this stride.
    default_score_stride = token_count;

    return AttentionQ16ApplyScoreScaleChecked(in_scores_q32,
                                              in_scores_capacity,
                                              token_count,
                                              default_score_stride,
                                              score_scale_q16,
                                              out_scores_q32,
                                              out_scores_capacity,
                                              default_score_stride);
}

I32 AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells)
{
    I64 default_score_stride;

    if (!in_scores_q32 || !out_scores_q32 || !out_last_in_base_index ||
        !out_last_out_base_index || !out_required_in_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-557 default-stride no-partial preflight helper:
    // score lanes are token-major at stride = token_count on both source and
    // destination; delegate to canonical no-partial preflight for all checked
    // overflow/span diagnostics.
    default_score_stride = token_count;

    return AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        default_score_stride,
        out_scores_q32,
        out_scores_capacity,
        default_score_stride,
        out_last_in_base_index,
        out_last_out_base_index,
        out_required_in_cells,
        out_required_out_cells);
}

I32 AttentionQ16ApplyScoreScaleCheckedNoPartialDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_score_stride;
    I64 last_in_base_index;
    I64 last_out_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-551/IQ-557 no-partial default-stride contract:
    // rows are token-major with one score lane every token_count cells.
    // Run default-stride no-partial preflight first so this wrapper exports
    // the same checked diagnostics path used by staged compositions.
    default_score_stride = token_count;

    status = AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        &last_in_base_index,
        &last_out_base_index,
        &required_in_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleCheckedNoPartial(in_scores_q32,
                                                       in_scores_capacity,
                                                       token_count,
                                                       default_score_stride,
                                                       score_scale_q16,
                                                       out_scores_q32,
                                                       out_scores_capacity,
                                                       default_score_stride);
}

I32 AttentionQ16ApplyScoreScaleRowsChecked(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 in_row_cells;
    I64 out_row_cells;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 row_base;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 0 || out_score_stride < 0 || row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    if (row_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 1 || out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Compute per-row active span for source and destination score layouts.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(out_row_cells,
                                       1,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Rows are laid out at fixed row_stride; each row must fit its active
    // token span so batched execution remains deterministic and auditable.
    if (row_stride < in_row_cells || row_stride < out_row_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       row_stride,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_in_cells,
                                       in_row_cells,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       out_row_cells,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-571: deterministic row-major dispatch with checked per-row score
    // scaling and explicit span guards before first caller-visible write.
    row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = AttentionQ16ApplyScoreScaleChecked(in_scores_q32 + row_base,
                                                    in_row_cells,
                                                    token_count,
                                                    in_score_stride,
                                                    score_scale_q16,
                                                    out_scores_q32 + row_base,
                                                    out_row_cells,
                                                    out_score_stride);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryAddI64Checked(row_base,
                                           row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
            return status;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_score_stride;
    I64 default_row_stride;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-588 default-stride contract for row-batched score scaling:
    // score lanes and row slabs both advance at token_count cells.
    default_row_stride = token_count;
    default_score_stride = token_count;

    return AttentionQ16ApplyScoreScaleRowsChecked(in_scores_q32,
                                                  in_scores_capacity,
                                                  row_count,
                                                  token_count,
                                                  default_score_stride,
                                                  default_score_stride,
                                                  default_row_stride,
                                                  score_scale_q16,
                                                  out_scores_q32,
                                                  out_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells)
{
    I64 default_score_stride;
    I64 default_row_stride;
    I64 in_row_cells;
    I64 out_row_cells;
    I64 required_in_cells;
    I64 required_out_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_required_in_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
    {
        *out_required_in_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    // IQ-572 default-stride no-partial diagnostics contract:
    // score rows are token-major, so score stride and row stride are both
    // token_count. This helper computes checked aggregate source/destination
    // spans for all rows before any staged no-partial score writes.
    default_score_stride = token_count;
    default_row_stride = token_count;

    if (default_score_stride < 1 || default_row_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       default_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       default_score_stride,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(out_row_cells,
                                       1,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (default_row_stride < in_row_cells || default_row_stride < out_row_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       default_row_stride,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_in_cells,
                                       in_row_cells,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       default_row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       out_row_cells,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_required_in_cells = required_in_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 default_score_stride;
    I64 default_row_stride;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 row_base;
    I64 row_index;
    I64 token_index;
    I64 stage_index;
    I64 out_index;
    I64 next_row_base;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        &required_in_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (staged_scores_capacity < required_stage_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    // No-partial contract: staging must be disjoint from live sources/sinks.
    if (staged_scores_q32 == in_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == out_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;

    default_score_stride = token_count;
    default_row_stride = token_count;

    status = AttentionQ16ApplyScoreScaleRowsChecked(in_scores_q32,
                                                    in_scores_capacity,
                                                    row_count,
                                                    token_count,
                                                    default_score_stride,
                                                    default_score_stride,
                                                    default_row_stride,
                                                    score_scale_q16,
                                                    staged_scores_q32,
                                                    staged_scores_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    // Commit only live token lanes. Row padding in destination remains intact.
    row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryMulI64Checked(row_index,
                                               token_count,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;
            status = AttentionTryAddI64Checked(stage_index,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;

            out_scores_q32[out_index] = staged_scores_q32[stage_index];
        }

        status = AttentionTryAddI64Checked(row_base,
                                           default_row_stride,
                                           &next_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        row_base = next_row_base;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowChecked(I64 *q_row_q16,
                                          I64 q_row_capacity,
                                          I64 *k_rows_q16,
                                          I64 k_rows_capacity,
                                          I64 token_count,
                                          I64 k_row_stride_q16,
                                          I64 head_dim,
                                          I64 score_scale_q16,
                                          I64 *out_scores_q32,
                                          I64 out_scores_capacity,
                                          I64 out_score_stride)
{
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_k_cells;
    I64 required_out_cells;
    I32 status;

    status = AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        &last_k_base_index,
        &last_out_base_index,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Stage 1: compute unscaled Q·K row in Q32 lanes.
    status = AttentionQ16ComputeQKDotRowChecked(q_row_q16,
                                                q_row_capacity,
                                                k_rows_q16,
                                                k_rows_capacity,
                                                token_count,
                                                k_row_stride_q16,
                                                head_dim,
                                                out_scores_q32,
                                                out_scores_capacity,
                                                out_score_stride);
    if (status != ATTN_Q16_OK)
        return status;

    // Stage 2: apply checked Q16 scale in-place over the same row layout so
    // callers get deterministic pre-softmax score buffers.
    return AttentionQ16ApplyScoreScaleChecked(out_scores_q32,
                                              out_scores_capacity,
                                              token_count,
                                              out_score_stride,
                                              score_scale_q16,
                                              out_scores_q32,
                                              out_scores_capacity,
                                              out_score_stride);
}

I32 AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 required_k_cells;
    I64 required_out_cells;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I32 status;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_last_k_base_index || !out_last_out_base_index ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || k_row_stride_q16 < 0 || head_dim < 0 ||
        out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (head_dim > q_row_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
    {
        *out_last_k_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_k_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    if (k_row_stride_q16 < head_dim)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Keep K/output span checks identical to QK-row checked core so wrappers
    // can preflight with exact success/error diagnostics before score writes.
    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_k_cells > k_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       k_row_stride_q16,
                                       &last_k_base_index);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &last_out_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_out_base_index,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_last_k_base_index = last_k_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 default_k_row_stride_q16;
    I64 default_out_score_stride;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_last_k_base_index || !out_last_out_base_index ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-559 default-stride no-partial diagnostics:
    // derive the same row-major strides used by staged no-partial wrappers
    // so callers can validate full K/output spans before score writes.
    default_k_row_stride_q16 = head_dim;
    default_out_score_stride = token_count;

    return AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_k_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    // IQ-563 no-partial row diagnostics are intentionally identical to the
    // checked row preflight contract. Staged wrappers use this helper before
    // any score writes so all required spans and base indices are validated
    // up-front with exact overflow/error behavior.
    return AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_k_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStride(I64 *q_row_q16,
                                                       I64 q_row_capacity,
                                                       I64 *k_rows_q16,
                                                       I64 k_rows_capacity,
                                                       I64 token_count,
                                                       I64 head_dim,
                                                       I64 score_scale_q16,
                                                       I64 *out_scores_q32,
                                                       I64 out_scores_capacity)
{
    I64 default_k_row_stride_q16;
    I64 default_out_score_stride;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-564 default row-major no-partial bridge contracts:
    // - K rows are tightly packed by head_dim lanes.
    // - Output score row is spaced by token_count lanes.
    // Delegate through the no-partial row helper so this default wrapper
    // preserves staged all-or-nothing semantics while keeping the same
    // stride derivation and checked overflow/span validation.
    default_k_row_stride_q16 = head_dim;
    default_out_score_stride = token_count;

    return AttentionQ16ComputeScaledQKRowCheckedNoPartial(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride);
}

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartial(I64 *q_row_q16,
                                                   I64 q_row_capacity,
                                                   I64 *k_rows_q16,
                                                   I64 k_rows_capacity,
                                                   I64 token_count,
                                                   I64 k_row_stride_q16,
                                                   I64 head_dim,
                                                   I64 score_scale_q16,
                                                   I64 *out_scores_q32,
                                                   I64 out_scores_capacity,
                                                   I64 out_score_stride)
{
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_k_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 required_out_cells;
    I64 token_index;
    I64 out_base;
    I64 *staged_scores_q32;
    I32 status;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || k_row_stride_q16 < 0 || head_dim < 0 ||
        out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        &last_k_base_index,
        &last_out_base_index,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!token_count)
        return ATTN_Q16_OK;

    // Stage into a dense contiguous token row first so failures never mutate
    // caller output. Commit to strided output only after full success.
    stage_cells = token_count;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ComputeScaledQKRowChecked(q_row_q16,
                                                   q_row_capacity,
                                                   k_rows_q16,
                                                   k_rows_capacity,
                                                   token_count,
                                                   k_row_stride_q16,
                                                   head_dim,
                                                   score_scale_q16,
                                                   staged_scores_q32,
                                                   stage_cells,
                                                   1);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        out_scores_q32[out_base] = staged_scores_q32[token_index];
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialDefaultStride(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_k_row_stride_q16;
    I64 default_out_score_stride;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-554 default-stride no-partial contract:
    // - key rows are tightly packed with head_dim cells per token.
    // - output scores are token-major with token_count stride.
    // Delegate to the checked no-partial core so staged all-or-nothing
    // semantics and overflow validation stay identical to explicit composition.
    default_k_row_stride_q16 = head_dim;
    default_out_score_stride = token_count;

    return AttentionQ16ComputeScaledQKRowCheckedNoPartial(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_last_q_base_index || !out_last_k_base_index ||
        !out_last_out_base_index || !out_required_q_cells ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (query_row_count > 0 && query_row_stride_q16 < head_dim)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count > 0 && token_count > 0 && out_row_stride < token_count)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!query_row_count)
    {
        *out_last_q_base_index = 0;
        *out_last_k_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_q_cells = 0;
        *out_required_k_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       query_row_stride_q16,
                                       &last_q_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_q_base_index,
                                       head_dim,
                                       &required_q_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_q_cells > q_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_k_cells > k_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
    {
        *out_last_q_base_index = last_q_base_index;
        *out_last_k_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_q_cells = required_q_cells;
        *out_required_k_cells = required_k_cells;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    status = AttentionTryMulI64Checked(token_count - 1,
                                       k_row_stride_q16,
                                       &last_k_base_index);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       out_row_stride,
                                       &last_out_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_out_base_index,
                                       token_count,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_last_q_base_index = last_q_base_index;
    *out_last_k_base_index = last_k_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_q_cells = required_q_cells;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I32 status;

    if (!out_last_q_base_index || !out_last_k_base_index ||
        !out_last_out_base_index || !out_required_q_cells ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    // IQ-587: preserve strict no-partial diagnostics semantics by staging
    // all preflight outputs locally and committing to caller pointers only on
    // full success.
    status = AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    *out_last_q_base_index = last_q_base_index;
    *out_last_k_base_index = last_k_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_q_cells = required_q_cells;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsChecked(I64 *q_rows_q16,
                                           I64 q_rows_capacity,
                                           I64 query_row_count,
                                           I64 query_row_stride_q16,
                                           I64 *k_rows_q16,
                                           I64 k_rows_capacity,
                                           I64 token_count,
                                           I64 k_row_stride_q16,
                                           I64 head_dim,
                                           I64 score_scale_q16,
                                           I64 *out_scores_q32,
                                           I64 out_scores_capacity,
                                           I64 out_row_stride)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 q_row_base;
    I64 out_row_base;
    I32 status;

    status = AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count)
        return ATTN_Q16_OK;

    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           query_row_stride_q16,
                                           &q_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           out_row_stride,
                                           &out_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionQ16ComputeScaledQKRowChecked(q_rows_q16 + q_row_base,
                                                       head_dim,
                                                       k_rows_q16,
                                                       k_rows_capacity,
                                                       token_count,
                                                       k_row_stride_q16,
                                                       head_dim,
                                                       score_scale_q16,
                                                       out_scores_q32 + out_row_base,
                                                       token_count,
                                                       1);
        if (status != ATTN_Q16_OK)
            return status;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartial(I64 *q_rows_q16,
                                                    I64 q_rows_capacity,
                                                    I64 query_row_count,
                                                    I64 query_row_stride_q16,
                                                    I64 *k_rows_q16,
                                                    I64 k_rows_capacity,
                                                    I64 token_count,
                                                    I64 k_row_stride_q16,
                                                    I64 head_dim,
                                                    I64 score_scale_q16,
                                                    I64 *out_scores_q32,
                                                    I64 out_scores_capacity,
                                                    I64 out_row_stride)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 row_index;
    I64 token_index;
    I64 row_base;
    I64 out_index;
    I64 *staged_scores_q32;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Empty query batches never touch output.
    if (!query_row_count)
        return ATTN_Q16_OK;

    // If there are no tokens, row cores do zero work and perform no writes.
    if (!token_count)
    {
        return AttentionQ16ComputeScaledQKRowsChecked(
            q_rows_q16,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_scores_q32,
            out_scores_capacity,
            out_row_stride);
    }

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Stage complete row-major score slab so any failure leaves caller output
    // untouched. RowsChecked owns all math/overflow checks.
    stage_cells = required_out_cells;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ComputeScaledQKRowsChecked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        stage_cells,
        out_row_stride);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    // Commit only live token cells (preserve caller padding between rows).
    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           out_row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
            {
                Free(staged_scores_q32);
                return status;
            }

            out_scores_q32[out_index] = staged_scores_q32[out_index];
        }
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_required_q_cells || !out_required_k_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-562 default-stride no-partial diagnostics contract:
    // - query rows are tightly packed by `head_dim` lanes.
    // - key rows are tightly packed by `head_dim` lanes.
    // - output rows are token-major with `token_count` stride.
    // This function computes all span diagnostics up-front so staged wrappers
    // can reject before any score writes to caller output buffers.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells);
    return status;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-560 default-stride batch contract:
    // - query rows are tightly packed head vectors.
    // - key rows are tightly packed head vectors.
    // - output rows are token-major with token_count row stride.
    // Delegate to the checked rows core so overflow and row-major
    // layout behavior are identical to explicit composition.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    return AttentionQ16ComputeScaledQKRowsChecked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStride(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-561 default-stride no-partial batch contract:
    // - query rows are tightly packed head vectors.
    // - key rows are tightly packed head vectors.
    // - output rows are token-major with token_count row stride.
    // Delegate to the strict no-partial rows core so staged all-or-nothing
    // write semantics and overflow checks exactly match explicit composition.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartial(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride);
}

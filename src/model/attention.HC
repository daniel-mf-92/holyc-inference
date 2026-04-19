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
    I64 total_in_cells;
    I64 total_out_cells;
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

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || in_score_stride < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
        return ATTN_Q16_OK;

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

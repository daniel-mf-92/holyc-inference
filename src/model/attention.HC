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

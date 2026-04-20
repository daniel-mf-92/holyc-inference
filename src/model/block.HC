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

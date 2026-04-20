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

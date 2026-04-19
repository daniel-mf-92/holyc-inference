// ffn.HC
// Integer-only Q16 SwiGLU helpers for HolyC inference.
//
// SwiGLU lane math:
//   out = silu(gate) * up
//   silu(x) = x * sigmoid(x)
//   sigmoid(x) = 1 / (1 + exp(-x))
//
// All values are Q16.16 in I64 lanes.

#define FFN_Q16_OK            0
#define FFN_Q16_ERR_NULL_PTR  1
#define FFN_Q16_ERR_BAD_PARAM 2
#define FFN_Q16_ERR_OVERFLOW  4

#define FFN_Q16_SHIFT 16
#define FFN_Q16_ONE   (1 << FFN_Q16_SHIFT)

#define FFN_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define FFN_I64_MIN_VALUE -0x8000000000000000

I32 FPQ16MulChecked(I64 a_q16, I64 b_q16, I64 *out_q16);
I32 FPQ16DivChecked(I64 num_q16, I64 den_q16, I64 *out_q16);
I32 FPQ16ExpClampToInputDomainChecked(I64 input_q16, I64 *out_clamped_q16);
I32 FPQ16ExpFromClampedInputChecked(I64 clamped_input_q16, I64 *out_exp_q16);

I32 FFNTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FFN_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > FFN_I64_MAX_VALUE - rhs)
        return FFN_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < FFN_I64_MIN_VALUE - rhs)
        return FFN_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return FFN_Q16_OK;
}

I32 FFNTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FFN_Q16_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return FFN_Q16_OK;
    }

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > (FFN_I64_MAX_VALUE / rhs))
                return FFN_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (rhs < (FFN_I64_MIN_VALUE / lhs))
                return FFN_Q16_ERR_OVERFLOW;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < (FFN_I64_MIN_VALUE / rhs))
                return FFN_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (lhs != 0 && rhs < (FFN_I64_MAX_VALUE / lhs))
                return FFN_Q16_ERR_OVERFLOW;
        }
    }

    *out_prod = lhs * rhs;
    return FFN_Q16_OK;
}

I32 FFNQ16SigmoidChecked(I64 gate_q16, I64 *out_sigmoid_q16)
{
    I64 neg_gate_q16;
    I64 clamped_neg_gate_q16;
    I64 exp_neg_gate_q16;
    I64 denom_q16;
    I32 status;

    if (!out_sigmoid_q16)
        return FFN_Q16_ERR_NULL_PTR;

    // Exact negate with explicit I64_MIN overflow guard.
    if (gate_q16 == FFN_I64_MIN_VALUE)
        return FFN_Q16_ERR_OVERFLOW;
    neg_gate_q16 = -gate_q16;

    // exp() path stays integer-only and saturating via exp-domain clamp.
    status = FPQ16ExpClampToInputDomainChecked(neg_gate_q16,
                                               &clamped_neg_gate_q16);
    if (status)
        return status;

    status = FPQ16ExpFromClampedInputChecked(clamped_neg_gate_q16,
                                             &exp_neg_gate_q16);
    if (status)
        return status;

    status = FFNTryAddI64Checked(FFN_Q16_ONE,
                                 exp_neg_gate_q16,
                                 &denom_q16);
    if (status)
        return status;

    return FPQ16DivChecked(FFN_Q16_ONE,
                           denom_q16,
                           out_sigmoid_q16);
}

I32 FFNQ16SiLUChecked(I64 gate_q16, I64 *out_silu_q16)
{
    I64 sigmoid_q16;
    I32 status;

    if (!out_silu_q16)
        return FFN_Q16_ERR_NULL_PTR;

    status = FFNQ16SigmoidChecked(gate_q16,
                                  &sigmoid_q16);
    if (status)
        return status;

    // silu(gate) = gate * sigmoid(gate), Q16 x Q16 -> Q16.
    return FPQ16MulChecked(gate_q16,
                           sigmoid_q16,
                           out_silu_q16);
}

I32 FFNQ16SwiGLULaneChecked(I64 gate_q16,
                            I64 up_q16,
                            I64 *out_lane_q16)
{
    I64 silu_q16;
    I32 status;

    if (!out_lane_q16)
        return FFN_Q16_ERR_NULL_PTR;

    status = FFNQ16SiLUChecked(gate_q16,
                               &silu_q16);
    if (status)
        return status;

    // SwiGLU output lane: silu(gate) * up.
    return FPQ16MulChecked(silu_q16,
                           up_q16,
                           out_lane_q16);
}

// Checked batched SwiGLU apply over strided Q16 lane arrays.
//
// Contracts:
// - lane_count >= 0
// - all capacities/strides >= 0
// - when lane_count > 0, all strides must be >= 1
// - capacity is measured in I64 lanes (not bytes)
//
// This helper validates source/destination spans before lane writes so index
// arithmetic cannot walk outside caller-provided buffers.
I32 FFNQ16SwiGLUApplyChecked(I64 *gate_q16,
                             I64 gate_capacity,
                             I64 gate_stride,
                             I64 *up_q16,
                             I64 up_capacity,
                             I64 up_stride,
                             I64 *out_q16,
                             I64 out_capacity,
                             I64 out_stride,
                             I64 lane_count)
{
    I64 lane_index;
    I64 last_lane;
    I64 required_gate_cells;
    I64 required_up_cells;
    I64 required_out_cells;
    I64 gate_base;
    I64 up_base;
    I64 out_base;
    I64 lane_out_q16;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (gate_stride < 0 || up_stride < 0 || out_stride < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return FFN_Q16_OK;

    if (gate_stride < 1 || up_stride < 1 || out_stride < 1)
        return FFN_Q16_ERR_BAD_PARAM;

    last_lane = lane_count - 1;

    status = FFNTryMulI64Checked(last_lane,
                                 gate_stride,
                                 &required_gate_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_gate_cells,
                                 1,
                                 &required_gate_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_lane,
                                 up_stride,
                                 &required_up_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_up_cells,
                                 1,
                                 &required_up_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_lane,
                                 out_stride,
                                 &required_out_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_out_cells,
                                 1,
                                 &required_out_cells);
    if (status)
        return status;

    if (required_gate_cells > gate_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_up_cells > up_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = FFNTryMulI64Checked(lane_index,
                                     gate_stride,
                                     &gate_base);
        if (status)
            return status;

        status = FFNTryMulI64Checked(lane_index,
                                     up_stride,
                                     &up_base);
        if (status)
            return status;

        status = FFNTryMulI64Checked(lane_index,
                                     out_stride,
                                     &out_base);
        if (status)
            return status;

        status = FFNQ16SwiGLULaneChecked(gate_q16[gate_base],
                                         up_q16[up_base],
                                         &lane_out_q16);
        if (status)
            return status;

        out_q16[out_base] = lane_out_q16;
    }

    return FFN_Q16_OK;
}

// Canonical default-capacity wrapper over FFNQ16SwiGLUApplyChecked.
//
// Default contracts:
// - gate/up/out spans are contiguous (stride = 1)
// - capacities are exactly lane_count for all buffers
// - all pointer/null and negative-lane guards are explicit here so call-sites
//   can use one predictable entrypoint for dense lane vectors
I32 FFNQ16SwiGLUApplyCheckedDefaultCapacity(I64 *gate_q16,
                                            I64 *up_q16,
                                            I64 *out_q16,
                                            I64 lane_count)
{
    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    return FFNQ16SwiGLUApplyChecked(gate_q16,
                                    lane_count,
                                    1,
                                    up_q16,
                                    lane_count,
                                    1,
                                    out_q16,
                                    lane_count,
                                    1,
                                    lane_count);
}

// Checked all-or-nothing wrapper over FFNQ16SwiGLUApplyChecked.
//
// Behavior:
// - validates output-span math up-front
// - computes into staging buffer first
// - commits only active output lanes on success
// - preserves out_q16 on any error (no partial caller-visible writes)
I32 FFNQ16SwiGLUApplyCheckedNoPartial(I64 *gate_q16,
                                      I64 gate_capacity,
                                      I64 gate_stride,
                                      I64 *up_q16,
                                      I64 up_capacity,
                                      I64 up_stride,
                                      I64 *out_q16,
                                      I64 out_capacity,
                                      I64 out_stride,
                                      I64 lane_count)
{
    I64 last_lane;
    I64 required_out_cells;
    I64 staging_bytes;
    I64 lane_index;
    I64 out_base;
    I64 *staging_out_q16;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (gate_stride < 0 || up_stride < 0 || out_stride < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return FFN_Q16_OK;

    if (out_stride < 1)
        return FFN_Q16_ERR_BAD_PARAM;

    last_lane = lane_count - 1;
    status = FFNTryMulI64Checked(last_lane,
                                 out_stride,
                                 &required_out_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_out_cells,
                                 1,
                                 &required_out_cells);
    if (status)
        return status;

    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(required_out_cells,
                                 8,
                                 &staging_bytes);
    if (status)
        return status;

    staging_out_q16 = MAlloc(staging_bytes);
    if (!staging_out_q16)
        return FFN_Q16_ERR_OVERFLOW;

    status = FFNQ16SwiGLUApplyChecked(gate_q16,
                                      gate_capacity,
                                      gate_stride,
                                      up_q16,
                                      up_capacity,
                                      up_stride,
                                      staging_out_q16,
                                      required_out_cells,
                                      out_stride,
                                      lane_count);
    if (status)
    {
        Free(staging_out_q16);
        return status;
    }

    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = FFNTryMulI64Checked(lane_index,
                                     out_stride,
                                     &out_base);
        if (status)
        {
            Free(staging_out_q16);
            return status;
        }

        out_q16[out_base] = staging_out_q16[out_base];
    }

    Free(staging_out_q16);
    return FFN_Q16_OK;
}

// Checked batched row-major SwiGLU apply composed from lane helper.
//
// Each row is a dense lane vector of length lane_count (lane stride = 1).
// Row strides are measured in I64 lanes between row bases.
I32 FFNQ16SwiGLUApplyRowsChecked(I64 *gate_q16,
                                 I64 gate_capacity,
                                 I64 gate_row_stride,
                                 I64 *up_q16,
                                 I64 up_capacity,
                                 I64 up_row_stride,
                                 I64 *out_q16,
                                 I64 out_capacity,
                                 I64 out_row_stride,
                                 I64 row_count,
                                 I64 lane_count)
{
    I64 last_row;
    I64 required_gate_cells;
    I64 required_up_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 gate_row_base;
    I64 up_row_base;
    I64 out_row_base;
    I64 next_gate_row_base;
    I64 next_up_row_base;
    I64 next_out_row_base;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (gate_row_stride < 0 || up_row_stride < 0 || out_row_stride < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    if (gate_row_stride < lane_count || up_row_stride < lane_count ||
        out_row_stride < lane_count)
        return FFN_Q16_ERR_BAD_PARAM;

    last_row = row_count - 1;

    status = FFNTryMulI64Checked(last_row,
                                 gate_row_stride,
                                 &required_gate_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_gate_cells,
                                 lane_count,
                                 &required_gate_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_row,
                                 up_row_stride,
                                 &required_up_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_up_cells,
                                 lane_count,
                                 &required_up_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_row,
                                 out_row_stride,
                                 &required_out_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_out_cells,
                                 lane_count,
                                 &required_out_cells);
    if (status)
        return status;

    if (required_gate_cells > gate_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_up_cells > up_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    // Deterministic row-major execution: row 0 -> row N-1,
    // each row writes exactly lane_count dense cells.
    gate_row_base = 0;
    up_row_base = 0;
    out_row_base = 0;

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = FFNQ16SwiGLUApplyChecked(gate_q16 + gate_row_base,
                                          lane_count,
                                          1,
                                          up_q16 + up_row_base,
                                          lane_count,
                                          1,
                                          out_q16 + out_row_base,
                                          lane_count,
                                          1,
                                          lane_count);
        if (status)
            return status;

        status = FFNTryAddI64Checked(gate_row_base,
                                     gate_row_stride,
                                     &next_gate_row_base);
        if (status)
            return status;

        status = FFNTryAddI64Checked(up_row_base,
                                     up_row_stride,
                                     &next_up_row_base);
        if (status)
            return status;

        status = FFNTryAddI64Checked(out_row_base,
                                     out_row_stride,
                                     &next_out_row_base);
        if (status)
            return status;

        gate_row_base = next_gate_row_base;
        up_row_base = next_up_row_base;
        out_row_base = next_out_row_base;
    }

    return FFN_Q16_OK;
}

// Checked all-or-nothing row-major wrapper over FFNQ16SwiGLUApplyRowsChecked.
//
// Behavior:
// - validates row/lane/stride output-span math up front
// - computes complete row batch into a staging buffer first
// - commits only live lane_count cells per row on success
// - preserves caller out_q16 on any failure (no partial writes)
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartial(I64 *gate_q16,
                                          I64 gate_capacity,
                                          I64 gate_row_stride,
                                          I64 *up_q16,
                                          I64 up_capacity,
                                          I64 up_row_stride,
                                          I64 *out_q16,
                                          I64 out_capacity,
                                          I64 out_row_stride,
                                          I64 row_count,
                                          I64 lane_count)
{
    I64 last_row;
    I64 required_out_cells;
    I64 staging_bytes;
    I64 row_index;
    I64 lane_index;
    I64 out_row_base;
    I64 next_out_row_base;
    I64 out_index;
    I64 *staging_out_q16;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (gate_row_stride < 0 || up_row_stride < 0 || out_row_stride < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    if (gate_row_stride < lane_count || up_row_stride < lane_count ||
        out_row_stride < lane_count)
        return FFN_Q16_ERR_BAD_PARAM;

    last_row = row_count - 1;
    status = FFNTryMulI64Checked(last_row,
                                 out_row_stride,
                                 &required_out_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_out_cells,
                                 lane_count,
                                 &required_out_cells);
    if (status)
        return status;

    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(required_out_cells,
                                 8,
                                 &staging_bytes);
    if (status)
        return status;

    staging_out_q16 = MAlloc(staging_bytes);
    if (!staging_out_q16)
        return FFN_Q16_ERR_OVERFLOW;

    status = FFNQ16SwiGLUApplyRowsChecked(gate_q16,
                                          gate_capacity,
                                          gate_row_stride,
                                          up_q16,
                                          up_capacity,
                                          up_row_stride,
                                          staging_out_q16,
                                          required_out_cells,
                                          out_row_stride,
                                          row_count,
                                          lane_count);
    if (status)
    {
        Free(staging_out_q16);
        return status;
    }

    out_row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FFNTryAddI64Checked(out_row_base,
                                         lane_index,
                                         &out_index);
            if (status)
            {
                Free(staging_out_q16);
                return status;
            }

            out_q16[out_index] = staging_out_q16[out_index];
        }

        status = FFNTryAddI64Checked(out_row_base,
                                     out_row_stride,
                                     &next_out_row_base);
        if (status)
        {
            Free(staging_out_q16);
            return status;
        }

        out_row_base = next_out_row_base;
    }

    Free(staging_out_q16);
    return FFN_Q16_OK;
}

// Deterministic default-stride row wrapper for SwiGLU batch apply.
//
// Default stride means each row is dense with no padding, so every row
// starts exactly lane_count cells after the previous row.
I32 FFNQ16SwiGLUApplyRowsCheckedDefaultStride(I64 *gate_q16,
                                              I64 gate_capacity,
                                              I64 *up_q16,
                                              I64 up_capacity,
                                              I64 *out_q16,
                                              I64 out_capacity,
                                              I64 row_count,
                                              I64 lane_count)
{
    I64 default_row_stride;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    default_row_stride = lane_count;

    return FFNQ16SwiGLUApplyRowsChecked(gate_q16,
                                        gate_capacity,
                                        default_row_stride,
                                        up_q16,
                                        up_capacity,
                                        default_row_stride,
                                        out_q16,
                                        out_capacity,
                                        default_row_stride,
                                        row_count,
                                        lane_count);
}

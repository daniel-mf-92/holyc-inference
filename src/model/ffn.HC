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

// Default-stride preflight helper for row-wise SwiGLU wrappers.
//
// Exposes exact checked diagnostics shared by default-stride wrappers:
// - default_row_stride (always lane_count)
// - last_row_index (row_count - 1 when rows are present)
// - required_{gate,up,out}_cells for capacity guards
I32 FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly(I64 gate_capacity,
                                                           I64 up_capacity,
                                                           I64 out_capacity,
                                                           I64 row_count,
                                                           I64 lane_count,
                                                           I64 *out_default_row_stride,
                                                           I64 *out_last_row_index,
                                                           I64 *out_required_gate_cells,
                                                           I64 *out_required_up_cells,
                                                           I64 *out_required_out_cells)
{
    I64 default_row_stride;
    I64 last_row_index;
    I64 required_gate_cells;
    I64 required_up_cells;
    I64 required_out_cells;
    I32 status;

    if (!out_default_row_stride || !out_last_row_index ||
        !out_required_gate_cells || !out_required_up_cells ||
        !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    default_row_stride = lane_count;

    if (!row_count || !lane_count)
    {
        *out_default_row_stride = default_row_stride;
        *out_last_row_index = 0;
        *out_required_gate_cells = 0;
        *out_required_up_cells = 0;
        *out_required_out_cells = 0;
        return FFN_Q16_OK;
    }

    last_row_index = row_count - 1;

    status = FFNTryMulI64Checked(last_row_index,
                                 default_row_stride,
                                 &required_gate_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_gate_cells,
                                 lane_count,
                                 &required_gate_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_row_index,
                                 default_row_stride,
                                 &required_up_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_up_cells,
                                 lane_count,
                                 &required_up_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_row_index,
                                 default_row_stride,
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

    *out_default_row_stride = default_row_stride;
    *out_last_row_index = last_row_index;
    *out_required_gate_cells = required_gate_cells;
    *out_required_up_cells = required_up_cells;
    *out_required_out_cells = required_out_cells;

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
    I64 last_row_index;
    I64 required_gate_cells;
    I64 required_up_cells;
    I64 required_out_cells;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    default_row_stride = lane_count;

    status = FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly(gate_capacity,
                                                                    up_capacity,
                                                                    out_capacity,
                                                                    row_count,
                                                                    lane_count,
                                                                    &default_row_stride,
                                                                    &last_row_index,
                                                                    &required_gate_cells,
                                                                    &required_up_cells,
                                                                    &required_out_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

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

// All-or-nothing default-stride row wrapper for SwiGLU no-partial batch apply.
//
// Default stride means each row is dense with no padding, so all source and
// destination row strides collapse to lane_count.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStride(I64 *gate_q16,
                                                        I64 gate_capacity,
                                                        I64 *up_q16,
                                                        I64 up_capacity,
                                                        I64 *out_q16,
                                                       I64 out_capacity,
                                                       I64 row_count,
                                                       I64 lane_count)
{
    I64 default_row_stride;
    I64 last_row_index;
    I64 required_gate_cells;
    I64 required_up_cells;
    I64 required_out_cells;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    default_row_stride = lane_count;

    status = FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly(gate_capacity,
                                                                    up_capacity,
                                                                    out_capacity,
                                                                    row_count,
                                                                    lane_count,
                                                                    &default_row_stride,
                                                                    &last_row_index,
                                                                    &required_gate_cells,
                                                                    &required_up_cells,
                                                                    &required_out_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartial(gate_q16,
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

// No-alloc required-bytes helper for default-stride no-partial SwiGLU
// commit pipelines.
//
// Computes checked diagnostics used by caller-managed staging paths:
// - required_stage_cells = row_count * lane_count
// - required_stage_bytes = required_stage_cells * sizeof(I64)
// - required_out_cells from default row-major span math
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(I64 row_count,
                                                                            I64 lane_count,
                                                                            I64 stage_cell_capacity,
                                                                            I64 *out_required_stage_cells,
                                                                            I64 *out_required_stage_bytes,
                                                                            I64 *out_required_out_cells)
{
    I64 default_row_stride;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0 || stage_cell_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
    {
        *out_required_stage_cells = 0;
        *out_required_stage_bytes = 0;
        *out_required_out_cells = 0;
        return FFN_Q16_OK;
    }

    // Stage footprint is one dense row-major slab of live lanes.
    status = FFNTryMulI64Checked(row_count,
                                 lane_count,
                                 &required_stage_cells);
    if (status)
        return status;

    // Convert stage cells to bytes with checked overflow math.
    status = FFNTryMulI64Checked(required_stage_cells,
                                 sizeof(I64),
                                 &required_stage_bytes);
    if (status)
        return status;

    // Default-stride destination rows are token/lane-major contiguous.
    default_row_stride = lane_count;
    status = FFNTryMulI64Checked(row_count - 1,
                                 default_row_stride,
                                 &required_out_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_out_cells,
                                 lane_count,
                                 &required_out_cells);
    if (status)
        return status;

    if (required_stage_cells > stage_cell_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return FFN_Q16_OK;
}

// Default-capacity wrapper for no-alloc required-bytes diagnostics.
//
// IQ-610 contract:
// - derive default_stage_cell_capacity = row_count * lane_count with checked
//   overflow math;
// - route diagnostics through canonical RequiredBytes helper so capacity,
//   required-byte, and out-span checks stay in one shared path.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity(
    I64 row_count,
    I64 lane_count,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 default_stage_cell_capacity;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(row_count,
                                 lane_count,
                                 &default_stage_cell_capacity);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        row_count,
        lane_count,
        default_stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells);
}

// Commit-capacity diagnostics helper for caller-staged default-stride
// no-alloc no-partial SwiGLU rows.
//
// IQ-608 contract:
// - derive canonical stage/out requirements through RequiredBytes helper;
// - validate commit scratch cell+byte capacities in one checked path;
// - expose exact required diagnostics for commit-only dispatchers.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
    I64 row_count,
    I64 lane_count,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_required_stage_cells;
    I64 commit_required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        &commit_required_stage_cells,
        &commit_required_stage_bytes,
        &required_out_cells);
    if (status)
        return status;

    if (commit_required_stage_bytes > commit_stage_byte_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    *out_commit_required_stage_cells = commit_required_stage_cells;
    *out_commit_required_stage_bytes = commit_required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return FFN_Q16_OK;
}

// Default commit-cell-capacity wrapper for no-alloc required-bytes commit
// diagnostics.
//
// IQ-614 contract:
// - derive commit_stage_cell_capacity = row_count * lane_count using checked
//   overflow math;
// - route all diagnostics through the canonical commit-capacity helper so
//   null/shape/capacity/error behavior stays identical to explicit composition.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityDefault(
    I64 row_count,
    I64 lane_count,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0 || commit_stage_byte_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(row_count,
                                 lane_count,
                                 &commit_stage_cell_capacity);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
}

// Caller-staged no-alloc preflight diagnostics for default-stride
// no-partial SwiGLU rows.
//
// Returns checked staging/output requirements used by no-alloc commit wrappers:
// - required_stage_cells = row_count * lane_count
// - required_stage_bytes = required_stage_cells * sizeof(I64)
// - required_out_cells from default-stride span preflight
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(I64 gate_capacity,
                                                                            I64 up_capacity,
                                                                            I64 out_capacity,
                                                                            I64 row_count,
                                                                            I64 lane_count,
                                                                            I64 stage_capacity,
                                                                            I64 *out_required_stage_cells,
                                                                            I64 *out_required_stage_bytes,
                                                                            I64 *out_required_out_cells)
{
    I64 default_row_stride;
    I64 last_row_index;
    I64 required_gate_cells;
    I64 required_up_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells_from_required_bytes;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (stage_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    default_row_stride = lane_count;

    status = FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly(gate_capacity,
                                                                    up_capacity,
                                                                    out_capacity,
                                                                    row_count,
                                                                    lane_count,
                                                                    &default_row_stride,
                                                                    &last_row_index,
                                                                    &required_gate_cells,
                                                                    &required_up_cells,
                                                                    &required_out_cells);
    if (status)
        return status;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        row_count,
        lane_count,
        stage_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells_from_required_bytes);
    if (status)
        return status;

    // Required-bytes helper and full preflight must agree on the exact live
    // default-stride destination span before no-alloc commit pipelines proceed.
    if (required_out_cells_from_required_bytes != required_out_cells)
        return FFN_Q16_ERR_BAD_PARAM;

    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return FFN_Q16_OK;
}

// Commit-only helper for caller-staged default-stride no-alloc no-partial
// SwiGLU rows.
//
// Performs checked two-pass copy semantics:
// 1) validate every `{row_base + lane_index}` source/destination index;
// 2) commit live lane cells only after full validation succeeds.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(I64 row_count,
                                                                         I64 lane_count,
                                                                         I64 *staging_out_q16,
                                                                         I64 staging_out_capacity,
                                                                         I64 stage_cell_capacity,
                                                                         I64 *out_q16,
                                                                         I64 out_capacity)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I64 row_index;
    I64 lane_index;
    I64 row_base;
    I64 stage_row_base;
    I64 out_index;
    I64 stage_index;
    I32 status;

    if (!staging_out_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_capacity < 0 || stage_cell_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    // Commit helper is destination-only: staging and output must remain
    // disjoint so validation pass cannot be affected by in-place writes.
    if (staging_out_q16 == out_q16)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        row_count,
        lane_count,
        stage_cell_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status)
        return status;

    if (required_stage_cells > staging_out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_stage_cells > stage_cell_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = FFNTryMulI64Checked(row_index,
                                     lane_count,
                                     &row_base);
        if (status)
            return status;

        status = FFNTryMulI64Checked(row_index,
                                     lane_count,
                                     &stage_row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FFNTryAddI64Checked(row_base,
                                         lane_index,
                                         &out_index);
            if (status)
                return status;
            if (out_index < 0 || out_index >= required_out_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (out_index >= out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;

            status = FFNTryAddI64Checked(stage_row_base,
                                         lane_index,
                                         &stage_index);
            if (status)
                return status;
            if (stage_index < 0 || stage_index >= required_stage_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (stage_index >= staging_out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;
        }
    }

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = FFNTryMulI64Checked(row_index,
                                     lane_count,
                                     &row_base);
        if (status)
            return status;

        status = FFNTryMulI64Checked(row_index,
                                     lane_count,
                                     &stage_row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FFNTryAddI64Checked(row_base,
                                         lane_index,
                                         &out_index);
            if (status)
                return status;
            if (out_index < 0 || out_index >= required_out_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (out_index >= out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;

            status = FFNTryAddI64Checked(stage_row_base,
                                         lane_index,
                                         &stage_index);
            if (status)
                return status;
            if (stage_index < 0 || stage_index >= required_stage_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (stage_index >= staging_out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;

            out_q16[out_index] = staging_out_q16[stage_index];
        }
    }

    return FFN_Q16_OK;
}

// Caller-staged no-allocation default-stride no-partial SwiGLU wrapper.
//
// All-or-nothing contract:
// - run default-stride preflight once;
// - evaluate every live lane into caller staging;
// - commit only after full success.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc(I64 *gate_q16,
                                                              I64 gate_capacity,
                                                              I64 *up_q16,
                                                              I64 up_capacity,
                                                              I64 *out_q16,
                                                              I64 out_capacity,
                                                              I64 row_count,
                                                              I64 lane_count,
                                                              I64 *staging_out_q16,
                                                              I64 staging_out_capacity)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I64 staging_capacity_bytes;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        staging_out_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    // Commit-capacity diagnostics already derive exact live output cells.
    // Reject undersized destination before running staged compute so no
    // caller staging lanes are touched on guaranteed commit failure.
    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!staging_out_q16)
        return FFN_Q16_ERR_NULL_PTR;
    if (staging_out_capacity < required_stage_cells)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &staging_capacity_bytes);
    if (status)
        return status;
    if (required_stage_bytes > staging_capacity_bytes)
        return FFN_Q16_ERR_BAD_PARAM;

    // No-partial guarantee requires staging to be disjoint from all
    // live input/output buffers used by the compute and commit phases.
    if (staging_out_q16 == gate_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == up_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == out_q16)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedDefaultStride(gate_q16,
                                                       gate_capacity,
                                                       up_q16,
                                                       up_capacity,
                                                       staging_out_q16,
                                                       staging_out_capacity,
                                                       row_count,
                                                       lane_count);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells,
        out_q16,
        out_capacity);
}

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(I64 *gate_q16,
                                                                             I64 gate_capacity,
                                                                             I64 *up_q16,
                                                                             I64 up_capacity,
                                                                             I64 *out_q16,
                                                                             I64 out_capacity,
                                                                             I64 row_count,
                                                                             I64 lane_count,
                                                                             I64 commit_stage_cell_capacity,
                                                                             I64 commit_stage_byte_capacity,
                                                                             I64 *staging_out_q16,
                                                                             I64 staging_out_capacity)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I64 staging_capacity_bytes;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0 ||
        staging_out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &staging_capacity_bytes);
    if (status)
        return status;

    // IQ-612 commit-capacity no-alloc contract:
    // 1) enforce caller commit scratch capacities through the shared
    //    required-bytes helper;
    // 2) run default-stride SwiGLU into caller staging slab;
    // 3) commit staged live cells atomically via no-partial commit-only.
    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    // Commit-capacity diagnostics already derive exact live output cells.
    // Reject undersized destination before staged compute so no staging
    // lanes are modified on guaranteed commit failure.
    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!staging_out_q16)
        return FFN_Q16_ERR_NULL_PTR;
    if (required_stage_cells > staging_out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return FFN_Q16_ERR_BAD_PARAM;

    // No-partial guarantee requires staging to be disjoint from all
    // source/destination buffers used by compute and commit phases.
    if (staging_out_q16 == gate_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == up_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == out_q16)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedDefaultStride(gate_q16,
                                                       gate_capacity,
                                                       up_q16,
                                                       up_capacity,
                                                       staging_out_q16,
                                                       staging_out_capacity,
                                                       row_count,
                                                       lane_count);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells,
        out_q16,
        out_capacity);
}

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity(I64 *gate_q16,
                                                                                             I64 gate_capacity,
                                                                                             I64 *up_q16,
                                                                                             I64 up_capacity,
                                                                                             I64 *out_q16,
                                                                                             I64 out_capacity,
                                                                                             I64 row_count,
                                                                                             I64 lane_count,
                                                                                             I64 *staging_out_q16,
                                                                                             I64 staging_out_capacity)
{
    I64 default_commit_stage_cell_capacity;
    I64 default_commit_stage_byte_capacity;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0 ||
        staging_out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    // IQ-620 default-capacity commit contract:
    // derive commit scratch limits from live row/lane geometry
    // (`row_count * lane_count`) and caller staging bytes
    // (`staging_out_capacity * sizeof(I64)`), then route all
    // no-partial semantics through the canonical commit-capacity wrapper.
    status = FFNTryMulI64Checked(row_count,
                                 lane_count,
                                 &default_commit_stage_cell_capacity);
    if (status)
        return status;

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &default_commit_stage_byte_capacity);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staging_out_q16,
        staging_out_capacity);
}

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity(I64 *gate_q16,
                                                                              I64 gate_capacity,
                                                                              I64 *up_q16,
                                                                              I64 up_capacity,
                                                                              I64 *out_q16,
                                                                              I64 out_capacity,
                                                                              I64 row_count,
                                                                              I64 lane_count,
                                                                              I64 *staging_out_q16,
                                                                              I64 staging_out_capacity)
{
    I64 default_stage_cell_capacity;
    I64 staging_capacity_bytes;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0 ||
        staging_out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(row_count,
                                 lane_count,
                                 &default_stage_cell_capacity);
    if (status)
        return status;

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &staging_capacity_bytes);
    if (status)
        return status;

    // IQ-609 default-capacity no-alloc contract:
    // derive stage capacity from the exact live default-stride slab
    // (`row_count * lane_count`) and validate diagnostics through
    // commit-capacity required-bytes helper before compute+commit.
    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        row_count,
        lane_count,
        default_stage_cell_capacity,
        staging_capacity_bytes,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    // Keep explicit commit-capacity guards in the commit path as local,
    // auditable invariants even though preflight already validated them.
    if (required_stage_cells > commit_stage_cell_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > commit_stage_byte_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!staging_out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (required_stage_cells > staging_out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return FFN_Q16_ERR_BAD_PARAM;

    if (staging_out_q16 == gate_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == up_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == out_q16)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsCheckedDefaultStride(gate_q16,
                                                       gate_capacity,
                                                       up_q16,
                                                       up_capacity,
                                                       staging_out_q16,
                                                       staging_out_capacity,
                                                       row_count,
                                                       lane_count);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells,
        out_q16,
        out_capacity);
}

// Caller-staged no-allocation strided no-partial SwiGLU wrapper with explicit
// commit-capacity diagnostics.
//
// Contracts:
// - `in_row_stride_q16` is shared by gate/up rows.
// - `out_row_stride_q16` controls both staged and committed row spacing.
// - Commit scratch bounds (`commit_stage_cell_capacity`,
//   `commit_stage_byte_capacity`) are enforced before staged compute.
// - Two-pass commit keeps strict no-partial writes on `out_q16`.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityPreflightOnly(I64 gate_capacity,
                                                                                     I64 up_capacity,
                                                                                     I64 out_capacity,
                                                                                     I64 row_count,
                                                                                     I64 lane_count,
                                                                                     I64 in_row_stride_q16,
                                                                                     I64 out_row_stride_q16,
                                                                                     I64 commit_stage_cell_capacity,
                                                                                     I64 commit_stage_byte_capacity,
                                                                                     I64 *out_required_in_cells,
                                                                                     I64 *out_required_stage_cells,
                                                                                     I64 *out_required_stage_bytes,
                                                                                     I64 *out_required_out_cells)
{
    I64 last_row;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I32 status;

    if (!out_required_in_cells || !out_required_stage_cells ||
        !out_required_stage_bytes || !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (in_row_stride_q16 < 0 || out_row_stride_q16 < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
    {
        *out_required_in_cells = 0;
        *out_required_stage_cells = 0;
        *out_required_stage_bytes = 0;
        *out_required_out_cells = 0;
        return FFN_Q16_OK;
    }

    if (in_row_stride_q16 < lane_count)
        return FFN_Q16_ERR_BAD_PARAM;
    if (out_row_stride_q16 < lane_count)
        return FFN_Q16_ERR_BAD_PARAM;

    last_row = row_count - 1;

    status = FFNTryMulI64Checked(last_row,
                                 in_row_stride_q16,
                                 &required_in_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_in_cells,
                                 lane_count,
                                 &required_in_cells);
    if (status)
        return status;

    status = FFNTryMulI64Checked(last_row,
                                 out_row_stride_q16,
                                 &required_out_cells);
    if (status)
        return status;
    status = FFNTryAddI64Checked(required_out_cells,
                                 lane_count,
                                 &required_out_cells);
    if (status)
        return status;

    required_stage_cells = required_out_cells;

    status = FFNTryMulI64Checked(required_stage_cells,
                                 sizeof(I64),
                                 &required_stage_bytes);
    if (status)
        return status;

    if (required_in_cells > gate_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_in_cells > up_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    if (required_stage_cells > commit_stage_cell_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > commit_stage_byte_capacity)
        return FFN_Q16_ERR_BAD_PARAM;

    *out_required_in_cells = required_in_cells;
    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return FFN_Q16_OK;
}

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacityPreflightOnly(I64 gate_capacity,
                                                                                                    I64 up_capacity,
                                                                                                    I64 out_capacity,
                                                                                                    I64 row_count,
                                                                                                    I64 lane_count,
                                                                                                    I64 in_row_stride_q16,
                                                                                                    I64 out_row_stride_q16,
                                                                                                    I64 staging_out_capacity,
                                                                                                    I64 *out_required_in_cells,
                                                                                                    I64 *out_commit_required_stage_cells,
                                                                                                    I64 *out_commit_required_stage_bytes,
                                                                                                    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I64 commit_stage_byte_capacity;
    I32 status;

    if (!out_required_in_cells || !out_commit_required_stage_cells ||
        !out_commit_required_stage_bytes || !out_required_out_cells)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0 ||
        staging_out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (in_row_stride_q16 < 0 || out_row_stride_q16 < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
    {
        commit_stage_cell_capacity = 0;
    }
    else
    {
        status = FFNTryMulI64Checked(row_count - 1,
                                     out_row_stride_q16,
                                     &commit_stage_cell_capacity);
        if (status)
            return status;

        status = FFNTryAddI64Checked(commit_stage_cell_capacity,
                                     lane_count,
                                     &commit_stage_cell_capacity);
        if (status)
            return status;
    }

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &commit_stage_byte_capacity);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityPreflightOnly(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_required_in_cells,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
}

// Caller-staged no-allocation strided no-partial SwiGLU wrapper with explicit
// commit-capacity diagnostics.
//
// Contracts:
// - `in_row_stride_q16` is shared by gate/up rows.
// - `out_row_stride_q16` controls both staged and committed row spacing.
// - Commit scratch bounds (`commit_stage_cell_capacity`,
//   `commit_stage_byte_capacity`) are enforced before staged compute.
// - Two-pass commit keeps strict no-partial writes on `out_q16`.
I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity(I64 *gate_q16,
                                                                       I64 gate_capacity,
                                                                       I64 *up_q16,
                                                                       I64 up_capacity,
                                                                       I64 *out_q16,
                                                                       I64 out_capacity,
                                                                       I64 row_count,
                                                                       I64 lane_count,
                                                                       I64 in_row_stride_q16,
                                                                       I64 out_row_stride_q16,
                                                                       I64 commit_stage_cell_capacity,
                                                                       I64 commit_stage_byte_capacity,
                                                                       I64 *staging_out_q16,
                                                                       I64 staging_out_capacity)
{
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 staging_capacity_bytes;
    I64 row_index;
    I64 lane_index;
    I64 out_row_base;
    I64 stage_row_base;
    I64 next_out_row_base;
    I64 next_stage_row_base;
    I64 out_index;
    I64 stage_index;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0 ||
        staging_out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &staging_capacity_bytes);
    if (status)
        return status;

    status = FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityPreflightOnly(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        &required_in_cells,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
        return FFN_Q16_OK;

    if (!staging_out_q16)
        return FFN_Q16_ERR_NULL_PTR;
    if (required_stage_cells > staging_out_capacity)
        return FFN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return FFN_Q16_ERR_BAD_PARAM;

    // Staging must remain disjoint from all source and destination arrays so
    // staged compute and commit-only copy cannot self-alias.
    if (staging_out_q16 == gate_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == up_q16)
        return FFN_Q16_ERR_BAD_PARAM;
    if (staging_out_q16 == out_q16)
        return FFN_Q16_ERR_BAD_PARAM;

    status = FFNQ16SwiGLUApplyRowsChecked(gate_q16,
                                          gate_capacity,
                                          in_row_stride_q16,
                                          up_q16,
                                          up_capacity,
                                          in_row_stride_q16,
                                          staging_out_q16,
                                          required_stage_cells,
                                          out_row_stride_q16,
                                          row_count,
                                          lane_count);
    if (status)
        return status;

    // Commit-only pass 1: validate every live out/stage index before writes.
    out_row_base = 0;
    stage_row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FFNTryAddI64Checked(out_row_base,
                                         lane_index,
                                         &out_index);
            if (status)
                return status;
            if (out_index < 0 || out_index >= required_out_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (out_index >= out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;

            status = FFNTryAddI64Checked(stage_row_base,
                                         lane_index,
                                         &stage_index);
            if (status)
                return status;
            if (stage_index < 0 || stage_index >= required_stage_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (stage_index >= staging_out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;
        }

        status = FFNTryAddI64Checked(out_row_base,
                                     out_row_stride_q16,
                                     &next_out_row_base);
        if (status)
            return status;

        status = FFNTryAddI64Checked(stage_row_base,
                                     out_row_stride_q16,
                                     &next_stage_row_base);
        if (status)
            return status;

        out_row_base = next_out_row_base;
        stage_row_base = next_stage_row_base;
    }

    // Commit-only pass 2: copy exactly live lane_count cells per row.
    out_row_base = 0;
    stage_row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FFNTryAddI64Checked(out_row_base,
                                         lane_index,
                                         &out_index);
            if (status)
                return status;
            if (out_index < 0 || out_index >= required_out_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (out_index >= out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;

            status = FFNTryAddI64Checked(stage_row_base,
                                         lane_index,
                                         &stage_index);
            if (status)
                return status;
            if (stage_index < 0 || stage_index >= required_stage_cells)
                return FFN_Q16_ERR_BAD_PARAM;
            if (stage_index >= staging_out_capacity)
                return FFN_Q16_ERR_BAD_PARAM;

            out_q16[out_index] = staging_out_q16[stage_index];
        }

        status = FFNTryAddI64Checked(out_row_base,
                                     out_row_stride_q16,
                                     &next_out_row_base);
        if (status)
            return status;

        status = FFNTryAddI64Checked(stage_row_base,
                                     out_row_stride_q16,
                                     &next_stage_row_base);
        if (status)
            return status;

        out_row_base = next_out_row_base;
        stage_row_base = next_stage_row_base;
    }

    return FFN_Q16_OK;
}

I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity(I64 *gate_q16,
                                                                                       I64 gate_capacity,
                                                                                       I64 *up_q16,
                                                                                       I64 up_capacity,
                                                                                       I64 *out_q16,
                                                                                       I64 out_capacity,
                                                                                       I64 row_count,
                                                                                       I64 lane_count,
                                                                                       I64 in_row_stride_q16,
                                                                                       I64 out_row_stride_q16,
                                                                                       I64 *staging_out_q16,
                                                                                       I64 staging_out_capacity)
{
    I64 commit_stage_cell_capacity;
    I64 commit_stage_byte_capacity;
    I32 status;

    if (!gate_q16 || !up_q16 || !out_q16)
        return FFN_Q16_ERR_NULL_PTR;

    if (gate_capacity < 0 || up_capacity < 0 || out_capacity < 0 ||
        staging_out_capacity < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FFN_Q16_ERR_BAD_PARAM;
    if (in_row_stride_q16 < 0 || out_row_stride_q16 < 0)
        return FFN_Q16_ERR_BAD_PARAM;

    // IQ-633 default-capacity strided commit contract:
    // derive commit scratch limits from live output geometry
    // (`required_out_cells = (row_count-1)*out_row_stride_q16 + lane_count`)
    // and caller staging storage bytes (`staging_out_capacity * sizeof(I64)`),
    // then route strict no-partial behavior through canonical commit-capacity
    // wrapper.
    if (!row_count || !lane_count)
    {
        commit_stage_cell_capacity = 0;
    }
    else
    {
        status = FFNTryMulI64Checked(row_count - 1,
                                     out_row_stride_q16,
                                     &commit_stage_cell_capacity);
        if (status)
            return status;

        status = FFNTryAddI64Checked(commit_stage_cell_capacity,
                                     lane_count,
                                     &commit_stage_cell_capacity);
        if (status)
            return status;
    }

    status = FFNTryMulI64Checked(staging_out_capacity,
                                 sizeof(I64),
                                 &commit_stage_byte_capacity);
    if (status)
        return status;

    return FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staging_out_q16,
        staging_out_capacity);
}

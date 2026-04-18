// q4_0_avx2.HC
// AVX2-shape Q4_0 dot kernel built in auditable HolyC first.
//
// This file keeps the exact lane ordering and signed accumulation semantics
// explicit so a later inline-asm AVX2 body can drop in without changing math.

#define Q4_0_AVX2_VALUES_PER_BLOCK 32
#define Q4_0_AVX2_PACKED_BYTES     16

#define Q4_0_AVX2_OK               0
#define Q4_0_AVX2_ERR_NULL_PTR     1
#define Q4_0_AVX2_ERR_BAD_LEN      2
#define Q4_0_AVX2_ERR_OVERFLOW     3

#define Q4_0_AVX2_I64_MAX          0x7FFFFFFFFFFFFFFF
#define Q4_0_AVX2_I64_MIN          (-0x7FFFFFFFFFFFFFFF - 1)

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_AVX2_PACKED_BYTES];
};

Bool Q4_0AVX2TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs == -1 && rhs == Q4_0_AVX2_I64_MIN)
        return FALSE;
    if (rhs == -1 && lhs == Q4_0_AVX2_I64_MIN)
        return FALSE;

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > Q4_0_AVX2_I64_MAX / rhs)
                return FALSE;
        }
        else
        {
            if (rhs < Q4_0_AVX2_I64_MIN / lhs)
                return FALSE;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < Q4_0_AVX2_I64_MIN / rhs)
                return FALSE;
        }
        else
        {
            if (lhs < Q4_0_AVX2_I64_MAX / rhs)
                return FALSE;
        }
    }

    *out_prod = lhs * rhs;
    return TRUE;
}

I64 Q4_0AVX2RoundShiftRightUnsigned(I64 value, I64 shift)
{
    I64 round_bias;

    if (shift <= 0)
        return value;

    round_bias = 1 << (shift - 1);
    return (value + round_bias) >> shift;
}

I64 Q4_0AVX2F16ToQ16(U16 fp16_bits)
{
    I64 sign_bit;
    I64 exponent_bits;
    I64 fraction_bits;
    I64 mantissa;
    I64 shift_amount;
    I64 magnitude_q16;

    sign_bit = (fp16_bits >> 15) & 1;
    exponent_bits = (fp16_bits >> 10) & 0x1F;
    fraction_bits = fp16_bits & 0x03FF;

    if (!exponent_bits)
    {
        if (!fraction_bits)
            return 0;

        magnitude_q16 = Q4_0AVX2RoundShiftRightUnsigned(fraction_bits, 8);
        if (sign_bit)
            return -magnitude_q16;
        return magnitude_q16;
    }

    if (exponent_bits == 0x1F)
    {
        if (sign_bit)
            return -0x3FFFFFFFFFFFFFFF;
        return 0x3FFFFFFFFFFFFFFF;
    }

    mantissa = 1024 + fraction_bits;
    shift_amount = exponent_bits - 9;

    if (shift_amount >= 0)
        magnitude_q16 = mantissa << shift_amount;
    else
        magnitude_q16 = Q4_0AVX2RoundShiftRightUnsigned(mantissa, -shift_amount);

    if (sign_bit)
        return -magnitude_q16;
    return magnitude_q16;
}

I64 Q4_0AVX2NibbleToSigned(U8 packed, Bool upper_nibble)
{
    I64 q_unsigned;

    if (upper_nibble)
        q_unsigned = (packed >> 4) & 0x0F;
    else
        q_unsigned = packed & 0x0F;

    return q_unsigned - 8;
}

I32 Q4_0Unpack32ToI16LanesAVX2(U8 *packed_q4, I64 packed_len, I16 *dst_i16, I64 dst_len)
{
    I64 byte_index;
    I64 lane_base;
    U8 packed;

    if (!packed_q4 || !dst_i16)
        return Q4_0_AVX2_ERR_NULL_PTR;

    if (packed_len < Q4_0_AVX2_PACKED_BYTES)
        return Q4_0_AVX2_ERR_BAD_LEN;
    if (dst_len < Q4_0_AVX2_VALUES_PER_BLOCK)
        return Q4_0_AVX2_ERR_BAD_LEN;

    // Lane contract mirrors AVX2 unpack intent:
    // - each byte stores two 4-bit quant values.
    // - dst_i16[2*i + 0] is low nibble lane, dst_i16[2*i + 1] is high nibble lane.
    // - each 4-bit lane is re-centered from [0..15] to signed [-8..7].
    for (byte_index = 0; byte_index < Q4_0_AVX2_PACKED_BYTES; ++byte_index)
    {
        packed = packed_q4[byte_index];
        lane_base = byte_index << 1;

        dst_i16[lane_base] = (I16)Q4_0AVX2NibbleToSigned(packed, FALSE);
        dst_i16[lane_base + 1] = (I16)Q4_0AVX2NibbleToSigned(packed, TRUE);
    }

    return Q4_0_AVX2_OK;
}

I32 Q4_0DotI16LanesAVX2(I16 *lhs_i16,
                        I64 lhs_len,
                        I16 *rhs_i16,
                        I64 rhs_len,
                        I64 *out_dot_q0)
{
    I64 lane;
    I64 total_q0;

    if (!lhs_i16 || !rhs_i16 || !out_dot_q0)
        return Q4_0_AVX2_ERR_NULL_PTR;

    if (lhs_len < Q4_0_AVX2_VALUES_PER_BLOCK)
        return Q4_0_AVX2_ERR_BAD_LEN;
    if (rhs_len < Q4_0_AVX2_VALUES_PER_BLOCK)
        return Q4_0_AVX2_ERR_BAD_LEN;

    // Signed lane accumulation path is intentionally direct and auditable:
    // dot_q0 = sum_{lane=0..31} lhs[lane] * rhs[lane].
    total_q0 = 0;
    for (lane = 0; lane < Q4_0_AVX2_VALUES_PER_BLOCK; ++lane)
        total_q0 += ((I64)lhs_i16[lane]) * ((I64)rhs_i16[lane]);

    *out_dot_q0 = total_q0;
    return Q4_0_AVX2_OK;
}

I32 Q4_0DotProductBlockQ32AVX2(Q4_0Block *lhs, Q4_0Block *rhs, I64 *out_dot_q32)
{
    I16 lhs_lanes_i16[Q4_0_AVX2_VALUES_PER_BLOCK];
    I16 rhs_lanes_i16[Q4_0_AVX2_VALUES_PER_BLOCK];
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 dot_q0;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_AVX2_ERR_NULL_PTR;

    status = Q4_0Unpack32ToI16LanesAVX2(lhs->qs,
                                        Q4_0_AVX2_PACKED_BYTES,
                                        lhs_lanes_i16,
                                        Q4_0_AVX2_VALUES_PER_BLOCK);
    if (status != Q4_0_AVX2_OK)
        return status;

    status = Q4_0Unpack32ToI16LanesAVX2(rhs->qs,
                                        Q4_0_AVX2_PACKED_BYTES,
                                        rhs_lanes_i16,
                                        Q4_0_AVX2_VALUES_PER_BLOCK);
    if (status != Q4_0_AVX2_OK)
        return status;

    status = Q4_0DotI16LanesAVX2(lhs_lanes_i16,
                                 Q4_0_AVX2_VALUES_PER_BLOCK,
                                 rhs_lanes_i16,
                                 Q4_0_AVX2_VALUES_PER_BLOCK,
                                 &dot_q0);
    if (status != Q4_0_AVX2_OK)
        return status;

    lhs_scale_q16 = Q4_0AVX2F16ToQ16(lhs->d_fp16);
    rhs_scale_q16 = Q4_0AVX2F16ToQ16(rhs->d_fp16);
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16;

    // Q4_0 nibble dot is Q0. Scales are Q16 each.
    // So final block contribution is Q32.
    *out_dot_q32 = scale_prod_q32 * dot_q0;
    return Q4_0_AVX2_OK;
}

I32 Q4_0DotProductBlocksQ32AVX2(Q4_0Block *lhs,
                                Q4_0Block *rhs,
                                I64 block_count,
                                I64 *out_dot_q32)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_AVX2_ERR_BAD_LEN;

    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DotProductBlockQ32AVX2(&lhs[block_index], &rhs[block_index], &block_dot_q32);
        if (status != Q4_0_AVX2_OK)
            return status;

        total_dot_q32 += block_dot_q32;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_AVX2_OK;
}

I32 Q4_0DotProductBlocksQ32AVX2Checked(
    Q4_0Block *lhs,
    I64 lhs_block_capacity,
    Q4_0Block *rhs,
    I64 rhs_block_capacity,
    I64 block_count,
    I64 *out_dot_q32)
{
    // IQ-286 checked wrapper contract:
    // - Null checks are explicit and return ERR_NULL_PTR.
    // - Capacities and block_count must be non-negative.
    // - Requested block_count must fit both provided extents.
    // - Delegate to AVX2-shape core kernel so math/bit semantics stay single-source.
    // - No partial write: out_dot_q32 is only modified when the delegated kernel succeeds.
    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_AVX2_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0)
        return Q4_0_AVX2_ERR_BAD_LEN;
    if (block_count < 0)
        return Q4_0_AVX2_ERR_BAD_LEN;

    if (block_count > lhs_block_capacity)
        return Q4_0_AVX2_ERR_BAD_LEN;
    if (block_count > rhs_block_capacity)
        return Q4_0_AVX2_ERR_BAD_LEN;

    return Q4_0DotProductBlocksQ32AVX2(lhs,
                                       rhs,
                                       block_count,
                                       out_dot_q32);
}

I32 Q4_0DotRowsAVX2Q32Checked(
    Q4_0Block *matrix_blocks,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q4_0Block *vec_blocks,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q32)
{
    I64 row_index;
    I64 required_matrix_blocks;
    I64 row_base;
    Q4_0Block *row_ptr;
    I64 row_dot_q32;
    I32 status;

    if (!matrix_blocks || !vec_blocks || !out_rows_q32)
        return Q4_0_AVX2_ERR_NULL_PTR;

    if (matrix_block_capacity < 0 || vec_block_capacity < 0)
        return Q4_0_AVX2_ERR_BAD_LEN;
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q4_0_AVX2_ERR_BAD_LEN;

    // Row i lives at matrix_blocks + i*row_stride_blocks.
    // Each row consumes vec_block_count active blocks for its dot.
    if (row_count > 0 && row_stride_blocks < vec_block_count)
        return Q4_0_AVX2_ERR_BAD_LEN;
    if (vec_block_count > vec_block_capacity)
        return Q4_0_AVX2_ERR_BAD_LEN;

    // Prove that the strided matrix window is representable and in-bounds.
    if (!Q4_0AVX2TryMulI64(row_count, row_stride_blocks, &required_matrix_blocks))
        return Q4_0_AVX2_ERR_OVERFLOW;
    if (required_matrix_blocks > matrix_block_capacity)
        return Q4_0_AVX2_ERR_BAD_LEN;

    for (row_index = 0; row_index < row_count; ++row_index)
    {
        if (!Q4_0AVX2TryMulI64(row_index, row_stride_blocks, &row_base))
            return Q4_0_AVX2_ERR_OVERFLOW;

        row_ptr = &matrix_blocks[row_base];
        status = Q4_0DotProductBlocksQ32AVX2Checked(row_ptr,
                                                    matrix_block_capacity - row_base,
                                                    vec_blocks,
                                                    vec_block_capacity,
                                                    vec_block_count,
                                                    &row_dot_q32);
        if (status != Q4_0_AVX2_OK)
            return status;

        out_rows_q32[row_index] = row_dot_q32;
    }

    return Q4_0_AVX2_OK;
}

I32 Q4_0DotProductBlocksQ32AVX2CheckedDefault(
    Q4_0Block *lhs,
    Q4_0Block *rhs,
    I64 block_count,
    I64 *out_dot_q32)
{
    // IQ-301 canonical default-capacity wrapper:
    // - Caller provides the active block span only.
    // - Capacity defaults to `block_count` on both sides.
    // - All null/bounds/no-partial-write contracts come from the checked core.
    return Q4_0DotProductBlocksQ32AVX2Checked(lhs,
                                              block_count,
                                              rhs,
                                              block_count,
                                              block_count,
                                              out_dot_q32);
}

// q4_0_q8_0_dot.HC
// Naive integer-only mixed Q4_0 x Q8_0 dot kernels for TempleOS.
//
// Math:
//   Q4_0 decode: x_i = d4 * (q4_i - 8)
//   Q8_0 decode: y_i = d8 * q8_i
//
// Integer HolyC form:
//   d4_q16   = Round(d4 * 2^16)
//   d8_q16   = Round(d8 * 2^16)
//   dot_q32  = Sum(x_q16 * y_q16)
//            = (d4_q16 * d8_q16) * Sum(q4_signed_i * q8_signed_i)
//
// dot_q32 keeps 32 fractional bits; caller may round to Q16 as needed.

#define Q4_0_VALUES_PER_BLOCK 32
#define Q4_0_PACKED_BYTES     16
#define Q8_0_VALUES_PER_BLOCK 32
#define Q8_0_PACKED_BYTES     32

#define Q4_0_Q8_0_OK               0
#define Q4_0_Q8_0_ERR_NULL_PTR     1
#define Q4_0_Q8_0_ERR_BAD_DST_LEN  2
#define Q4_0_Q8_0_ERR_OVERFLOW     3
#define Q4_0_Q8_0_I64_MAX          0x7FFFFFFFFFFFFFFF
#define Q4_0_Q8_0_I64_MIN          (-0x7FFFFFFFFFFFFFFF - 1)
#define Q4_0_Q8_0_AVX2_LANES       32
#define Q4_0_Q8_0_AVX2_PAIR_COUNT  16

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_PACKED_BYTES];
};

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q8_0_PACKED_BYTES];
};

I64 Q4_0_Q8_0RoundShiftRightUnsigned(I64 value, I64 shift)
{
    I64 round_bias;

    if (shift <= 0)
        return value;

    round_bias = 1 << (shift - 1);
    return (value + round_bias) >> shift;
}

I64 Q4_0_Q8_0RoundShiftRightSigned(I64 value, I64 shift)
{
    if (shift <= 0)
        return value;

    if (value >= 0)
        return Q4_0_Q8_0RoundShiftRightUnsigned(value, shift);

    return -Q4_0_Q8_0RoundShiftRightUnsigned(-value, shift);
}

I64 Q4_0_Q8_0F16ToQ16(U16 fp16_bits)
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

        magnitude_q16 = Q4_0_Q8_0RoundShiftRightUnsigned(fraction_bits, 8);
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
        magnitude_q16 = Q4_0_Q8_0RoundShiftRightUnsigned(mantissa, -shift_amount);

    if (sign_bit)
        return -magnitude_q16;
    return magnitude_q16;
}

I64 Q4_0_Q8_0NibbleToSigned(U8 packed, Bool upper_nibble)
{
    I64 q_unsigned;

    if (upper_nibble)
        q_unsigned = (packed >> 4) & 0x0F;
    else
        q_unsigned = packed & 0x0F;

    return q_unsigned - 8;
}

Bool Q4_0Q8_0TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (rhs > 0 && lhs > Q4_0_Q8_0_I64_MAX - rhs)
        return FALSE;
    if (rhs < 0 && lhs < Q4_0_Q8_0_I64_MIN - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool Q4_0Q8_0TryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs > Q4_0_Q8_0_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q4_0Q8_0TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs == -1 && rhs == Q4_0_Q8_0_I64_MIN)
        return FALSE;
    if (rhs == -1 && lhs == Q4_0_Q8_0_I64_MIN)
        return FALSE;

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > Q4_0_Q8_0_I64_MAX / rhs)
                return FALSE;
        }
        else
        {
            if (rhs < Q4_0_Q8_0_I64_MIN / lhs)
                return FALSE;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < Q4_0_Q8_0_I64_MIN / rhs)
                return FALSE;
        }
        else
        {
            if (lhs < Q4_0_Q8_0_I64_MAX / rhs)
                return FALSE;
        }
    }

    *out_prod = lhs * rhs;
    return TRUE;
}

I32 Q4_0Q8_0PackQ4BlockToI16LanesAVX2(Q4_0Block *src_q4, I16 *dst_i16, I64 dst_len)
{
    I64 byte_index;
    U8 packed_q4;

    if (!src_q4 || !dst_i16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (dst_len < Q4_0_Q8_0_AVX2_LANES)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // AVX2-shape lane contract for mixed Q4_0 blocks:
    //   lane[2*i]   = low nibble signed value from qs[i]
    //   lane[2*i+1] = high nibble signed value from qs[i]
    // This preserves ggml nibble ordering exactly while widening to I16.
    for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
    {
        packed_q4 = src_q4->qs[byte_index];
        dst_i16[byte_index << 1] = (I16)Q4_0_Q8_0NibbleToSigned(packed_q4, FALSE);
        dst_i16[(byte_index << 1) + 1] = (I16)Q4_0_Q8_0NibbleToSigned(packed_q4, TRUE);
    }

    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0PackQ8BlockToI16LanesAVX2(Q8_0Block *src_q8, I16 *dst_i16, I64 dst_len)
{
    I64 lane;

    if (!src_q8 || !dst_i16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (dst_len < Q4_0_Q8_0_AVX2_LANES)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // Widen each signed 8-bit Q8 lane to signed I16 lane in-place order.
    for (lane = 0; lane < Q4_0_Q8_0_AVX2_LANES; ++lane)
        dst_i16[lane] = (I16)src_q8->qs[lane];

    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0MulI16LanesToI32PairsAVX2(
    I16 *lhs_i16,
    I64 lhs_len,
    I16 *rhs_i16,
    I64 rhs_len,
    I32 *out_pairs_i32,
    I64 out_len)
{
    I64 pair_index;
    I64 lane_base;

    if (!lhs_i16 || !rhs_i16 || !out_pairs_i32)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (lhs_len < Q4_0_Q8_0_AVX2_LANES || rhs_len < Q4_0_Q8_0_AVX2_LANES)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (out_len < Q4_0_Q8_0_AVX2_PAIR_COUNT)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // Mirrors AVX2 `vpmaddwd`: multiply adjacent I16 pairs, then add each pair.
    for (pair_index = 0; pair_index < Q4_0_Q8_0_AVX2_PAIR_COUNT; ++pair_index)
    {
        lane_base = pair_index << 1;
        out_pairs_i32[pair_index] =
            lhs_i16[lane_base] * rhs_i16[lane_base] +
            lhs_i16[lane_base + 1] * rhs_i16[lane_base + 1];
    }

    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0HSumI32PairsAVX2(I32 *pairs_i32, I64 pair_len, I64 *out_sum_i64)
{
    I64 pair_index;
    I64 sum_i64;
    I64 term;

    if (!pairs_i32 || !out_sum_i64)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (pair_len < Q4_0_Q8_0_AVX2_PAIR_COUNT)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    sum_i64 = 0;
    for (pair_index = 0; pair_index < Q4_0_Q8_0_AVX2_PAIR_COUNT; ++pair_index)
    {
        term = (I64)pairs_i32[pair_index];
        if (!Q4_0Q8_0TryAddI64(sum_i64, term, &sum_i64))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    *out_sum_i64 = sum_i64;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotI16LanesAVX2(I16 *lhs_i16, I64 lhs_len, I16 *rhs_i16, I64 rhs_len, I64 *out_dot_i64)
{
    I32 pair_sums_i32[Q4_0_Q8_0_AVX2_PAIR_COUNT];
    I32 status;

    if (!lhs_i16 || !rhs_i16 || !out_dot_i64)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    status = Q4_0Q8_0MulI16LanesToI32PairsAVX2(
        lhs_i16,
        lhs_len,
        rhs_i16,
        rhs_len,
        pair_sums_i32,
        Q4_0_Q8_0_AVX2_PAIR_COUNT);
    if (status != Q4_0_Q8_0_OK)
        return status;

    return Q4_0Q8_0HSumI32PairsAVX2(pair_sums_i32,
                                    Q4_0_Q8_0_AVX2_PAIR_COUNT,
                                    out_dot_i64);
}

I32 Q4_0Q8_0DotBlocksAVX2Q32Checked(
    Q4_0Block *lhs_q4,
    Q8_0Block *rhs_q8,
    I64 block_count,
    I64 *out_dot_q32)
{
    I64 block_index;
    I16 lhs_q4_lanes_i16[Q4_0_Q8_0_AVX2_LANES];
    I16 rhs_q8_lanes_i16[Q4_0_Q8_0_AVX2_LANES];
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 q_dot_q0;
    I64 scale_prod_q32;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // IQ-115 mixed AVX2-shape contract:
    // - Expand Q4_0 nibbles and Q8_0 bytes to 32 signed I16 lanes.
    // - Dot those lanes with pairwise-multiply + horizontal reduction.
    // - Apply Q16 scales to produce block Q32, then checked-add across blocks.
    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0Q8_0PackQ4BlockToI16LanesAVX2(&lhs_q4[block_index],
                                                   lhs_q4_lanes_i16,
                                                   Q4_0_Q8_0_AVX2_LANES);
        if (status != Q4_0_Q8_0_OK)
            return status;

        status = Q4_0Q8_0PackQ8BlockToI16LanesAVX2(&rhs_q8[block_index],
                                                   rhs_q8_lanes_i16,
                                                   Q4_0_Q8_0_AVX2_LANES);
        if (status != Q4_0_Q8_0_OK)
            return status;

        status = Q4_0Q8_0DotI16LanesAVX2(lhs_q4_lanes_i16,
                                         Q4_0_Q8_0_AVX2_LANES,
                                         rhs_q8_lanes_i16,
                                         Q4_0_Q8_0_AVX2_LANES,
                                         &q_dot_q0);
        if (status != Q4_0_Q8_0_OK)
            return status;

        lhs_scale_q16 = Q4_0_Q8_0F16ToQ16(lhs_q4[block_index].d_fp16);
        rhs_scale_q16 = Q4_0_Q8_0F16ToQ16(rhs_q8[block_index].d_fp16);

        if (!Q4_0Q8_0TryMulI64(lhs_scale_q16, rhs_scale_q16, &scale_prod_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64(scale_prod_q32, q_dot_q0, &block_dot_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(total_dot_q32, block_dot_q32, &total_dot_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0_Q8_0DotBlocksAVX2Q32Checked(Q4_0Block *lhs_q4,
                                     Q8_0Block *rhs_q8,
                                     I64 block_count,
                                     I64 *out_dot_q32)
{
    return Q4_0Q8_0DotBlocksAVX2Q32Checked(lhs_q4,
                                           rhs_q8,
                                           block_count,
                                           out_dot_q32);
}

I32 Q4_0_Q8_0DotProductBlockQ32(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 *out_dot_q32)
{
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 byte_index;
    I64 q_dot_q0;
    U8 packed_q4;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    lhs_scale_q16 = Q4_0_Q8_0F16ToQ16(lhs_q4->d_fp16);
    rhs_scale_q16 = Q4_0_Q8_0F16ToQ16(rhs_q8->d_fp16);
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16;

    // q_dot_q0 = Sum(q4_signed_i * q8_signed_i).
    // Q4 stores two values per byte, so map each nibble to one Q8 lane.
    q_dot_q0 = 0;
    for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
    {
        packed_q4 = lhs_q4->qs[byte_index];

        q_dot_q0 += Q4_0_Q8_0NibbleToSigned(packed_q4, FALSE) * rhs_q8->qs[(byte_index << 1)];
        q_dot_q0 += Q4_0_Q8_0NibbleToSigned(packed_q4, TRUE) * rhs_q8->qs[(byte_index << 1) + 1];
    }

    *out_dot_q32 = scale_prod_q32 * q_dot_q0;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotProductBlocksQ32(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *out_dot_q32);

I32 Q4_0_Q8_0DotProductBlocksQ32(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *out_dot_q32)
{
    return Q4_0Q8_0DotProductBlocksQ32(lhs_q4, rhs_q8, block_count, out_dot_q32);
}

I32 Q4_0Q8_0DotRowBlocksQ16(Q4_0Block *row_blocks_q4, Q8_0Block *vec_blocks_q8, I64 block_count, I64 *out_dot_q16);

I32 Q4_0_Q8_0DotRowBlocksQ16(Q4_0Block *row_blocks_q4, Q8_0Block *vec_blocks_q8, I64 block_count, I64 *out_dot_q16)
{
    return Q4_0Q8_0DotRowBlocksQ16(row_blocks_q4, vec_blocks_q8, block_count, out_dot_q16);
}

I32 Q4_0Q8_0DotProductBlocksQ32(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *out_dot_q32)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0_Q8_0DotProductBlockQ32(&lhs_q4[block_index], &rhs_q8[block_index], &block_dot_q32);
        if (status != Q4_0_Q8_0_OK)
            return status;

        // Keep the widened Q32 accumulation explicit and checked.
        // For realistic LLaMA dimensions this stays far inside I64,
        // but we still hard-fail if a hostile input tries to overflow.
        if (!Q4_0Q8_0TryAddI64(total_dot_q32, block_dot_q32, &total_dot_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotProductBlocksQ32ToQ16(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *out_dot_q16)
{
    I64 total_dot_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // Mixed-kernel WS4 matmul callers can keep the full dot in I64 Q32,
    // then round once when handing the lane back to Q16 accumulation.
    status = Q4_0Q8_0DotProductBlocksQ32(lhs_q4, rhs_q8, block_count, &total_dot_q32);
    if (status != Q4_0_Q8_0_OK)
        return status;

    *out_dot_q16 = Q4_0_Q8_0DotQ32ToQ16(total_dot_q32);
    return Q4_0_Q8_0_OK;
}

I32 Q4_0_Q8_0DotProductBlocksQ16Accumulate(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *io_accum_q16)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_q16;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !io_accum_q16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    total_q16 = *io_accum_q16;

    // Matmul callers accumulate rows in Q16 lanes.
    // Keep each block explicit and auditable:
    //   block_q32 -> rounded block_q16 -> add to running accumulator.
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0_Q8_0DotProductBlockQ32(&lhs_q4[block_index], &rhs_q8[block_index], &block_dot_q32);
        if (status != Q4_0_Q8_0_OK)
            return status;

        total_q16 += Q4_0_Q8_0DotQ32ToQ16(block_dot_q32);
    }

    *io_accum_q16 = total_q16;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0_Q8_0DotProductBlocksQ16AccumulateChecked(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *io_accum_q16)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 block_dot_q16;
    I64 total_q16;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !io_accum_q16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    total_q16 = *io_accum_q16;

    // Checked variant for hostile inputs:
    //   block Q32 dot -> rounded Q16 -> checked add into accumulator lane.
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0_Q8_0DotProductBlockQ32(&lhs_q4[block_index], &rhs_q8[block_index], &block_dot_q32);
        if (status != Q4_0_Q8_0_OK)
            return status;

        block_dot_q16 = Q4_0_Q8_0DotQ32ToQ16(block_dot_q32);
        if (!Q4_0Q8_0TryAddI64(total_q16, block_dot_q16, &total_q16))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    *io_accum_q16 = total_q16;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotRowBlocksQ16(Q4_0Block *row_blocks_q4, Q8_0Block *vec_blocks_q8, I64 block_count, I64 *out_dot_q16)
{
    I64 accum_q16;
    I32 status;

    if (!row_blocks_q4 || !vec_blocks_q8 || !out_dot_q16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // WS4 row kernels consume Q16 lanes directly.
    // Keep each 32-value block contribution visible: block Q32 dot -> rounded Q16 -> sum.
    // This is intentionally different from full-dot single-rounding helper Q32ToQ16.
    accum_q16 = 0;
    status = Q4_0_Q8_0DotProductBlocksQ16Accumulate(row_blocks_q4, vec_blocks_q8, block_count, &accum_q16);
    if (status != Q4_0_Q8_0_OK)
        return status;

    *out_dot_q16 = accum_q16;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotRowsQ16MatrixVector(Q4_0Block *matrix_blocks_q4, I64 row_count, I64 row_stride_blocks, Q8_0Block *vec_blocks_q8, I64 vec_block_count, I64 *out_rows_q16)
{
    I64 row_index;
    I64 row_dot_q16;
    Q4_0Block *row_ptr_q4;
    I32 status;

    if (!matrix_blocks_q4 || !vec_blocks_q8 || !out_rows_q16)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // Each matrix row starts at:
    //   matrix_blocks_q4 + row_index * row_stride_blocks
    // and consumes vec_block_count contiguous mixed-dot blocks.
    // Enforce stride >= active vector span so row windows stay explicit and auditable.
    if (row_count > 0 && row_stride_blocks < vec_block_count)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    for (row_index = 0; row_index < row_count; ++row_index)
    {
        row_ptr_q4 = &matrix_blocks_q4[row_index * row_stride_blocks];
        status = Q4_0Q8_0DotRowBlocksQ16(row_ptr_q4, vec_blocks_q8, vec_block_count, &row_dot_q16);
        if (status != Q4_0_Q8_0_OK)
            return status;

        out_rows_q16[row_index] = row_dot_q16;
    }

    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotRowsQ16MatrixVectorChecked(
    Q4_0Block *matrix_blocks_q4,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks_q8,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q16)
{
    I64 row_index;
    I64 row_dot_q16;
    I64 required_matrix_blocks;
    I64 row_base;
    Q4_0Block *row_ptr_q4;
    I32 status;

    if (!matrix_blocks_q4 || !vec_blocks_q8 || !out_rows_q16)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (matrix_block_capacity < 0 || vec_block_capacity < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (row_count > 0 && row_stride_blocks < vec_block_count)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (vec_block_count > vec_block_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // Capacity guards:
    //   required_matrix_blocks = row_count * row_stride_blocks
    // must be representable and within provided matrix extent.
    if (!Q4_0Q8_0TryMulI64NonNeg(row_count, row_stride_blocks, &required_matrix_blocks))
        return Q4_0_Q8_0_ERR_OVERFLOW;
    if (required_matrix_blocks > matrix_block_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    for (row_index = 0; row_index < row_count; ++row_index)
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(row_index, row_stride_blocks, &row_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        row_ptr_q4 = &matrix_blocks_q4[row_base];
        row_dot_q16 = 0;
        status = Q4_0_Q8_0DotProductBlocksQ16AccumulateChecked(row_ptr_q4, vec_blocks_q8, vec_block_count, &row_dot_q16);
        if (status != Q4_0_Q8_0_OK)
            return status;

        out_rows_q16[row_index] = row_dot_q16;
    }

    return Q4_0_Q8_0_OK;
}

I32 Q4_0_Q8_0DotRowsQ16MatrixVectorChecked(
    Q4_0Block *matrix_blocks_q4,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks_q8,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q16)
{
    return Q4_0Q8_0DotRowsQ16MatrixVectorChecked(
        matrix_blocks_q4,
        matrix_block_capacity,
        row_count,
        row_stride_blocks,
        vec_blocks_q8,
        vec_block_capacity,
        vec_block_count,
        out_rows_q16);
}

I32 Q4_0_Q8_0DotRowsQ16MatrixVector(Q4_0Block *matrix_blocks_q4, I64 row_count, I64 row_stride_blocks, Q8_0Block *vec_blocks_q8, I64 vec_block_count, I64 *out_rows_q16)
{
    return Q4_0Q8_0DotRowsQ16MatrixVector(matrix_blocks_q4, row_count, row_stride_blocks, vec_blocks_q8, vec_block_count, out_rows_q16);
}

I64 Q4_0_Q8_0DotQ32ToQ16(I64 dot_q32)
{
    return Q4_0_Q8_0RoundShiftRightSigned(dot_q32, 16);
}

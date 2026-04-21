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

I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayDotOne(
    Q4_0Block *lhs_q4,
    I64 lhs_base_block,
    I64 lhs_block_stride,
    Q8_0Block *rhs_q8,
    I64 rhs_base_block,
    I64 rhs_block_stride,
    I64 blocks_per_dot,
    I64 *out_dot_q32)
{
    I64 block_idx;
    I64 lhs_index;
    I64 rhs_index;
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 q_dot_q0;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I64 byte_index;
    I64 low_term_q0;
    I64 high_term_q0;
    U8 packed_q4;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;
    if (lhs_base_block < 0 || rhs_base_block < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_block_stride <= 0 || rhs_block_stride <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (blocks_per_dot < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    total_dot_q32 = 0;
    for (block_idx = 0; block_idx < blocks_per_dot; ++block_idx)
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(block_idx, lhs_block_stride, &lhs_index))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(lhs_base_block, lhs_index, &lhs_index))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryMulI64NonNeg(block_idx, rhs_block_stride, &rhs_index))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(rhs_base_block, rhs_index, &rhs_index))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        lhs_scale_q16 = Q4_0_Q8_0F16ToQ16(lhs_q4[lhs_index].d_fp16);
        rhs_scale_q16 = Q4_0_Q8_0F16ToQ16(rhs_q8[rhs_index].d_fp16);
        if (!Q4_0Q8_0TryMulI64(lhs_scale_q16, rhs_scale_q16, &scale_prod_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        q_dot_q0 = 0;
        for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
        {
            packed_q4 = lhs_q4[lhs_index].qs[byte_index];

            low_term_q0 = Q4_0_Q8_0NibbleToSigned(packed_q4, FALSE) * rhs_q8[rhs_index].qs[(byte_index << 1)];
            high_term_q0 = Q4_0_Q8_0NibbleToSigned(packed_q4, TRUE) * rhs_q8[rhs_index].qs[(byte_index << 1) + 1];

            if (!Q4_0Q8_0TryAddI64(q_dot_q0, low_term_q0, &q_dot_q0))
                return Q4_0_Q8_0_ERR_OVERFLOW;
            if (!Q4_0Q8_0TryAddI64(q_dot_q0, high_term_q0, &q_dot_q0))
                return Q4_0_Q8_0_ERR_OVERFLOW;
        }

        if (!Q4_0Q8_0TryMulI64(scale_prod_q32, q_dot_q0, &block_dot_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(total_dot_q32, block_dot_q32, &total_dot_q32))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_Q8_0_OK;
}

// Checked mixed Q4_0 x Q8_0 array dot kernel with strict no-partial semantics.
//
// Array layout:
//   lhs_pair_base = pair_idx * lhs_pair_stride_blocks
//   rhs_pair_base = pair_idx * rhs_pair_stride_blocks
//   lhs_block_idx = lhs_pair_base + block_idx * lhs_block_stride
//   rhs_block_idx = rhs_pair_base + block_idx * rhs_block_stride
//
// Each output lane computes one Q32 mixed dot across `blocks_per_dot` blocks.
// The function preflights every pair before publishing any output so a hard
// fault never commits partial output rows.
I32 Q4_0Q8_0DotQ32CheckedNoPartialArray(
    Q4_0Block *lhs_q4,
    I64 lhs_block_capacity,
    I64 lhs_pair_stride_blocks,
    I64 lhs_block_stride,
    Q8_0Block *rhs_q8,
    I64 rhs_block_capacity,
    I64 rhs_pair_stride_blocks,
    I64 rhs_block_stride,
    I64 pair_count,
    I64 blocks_per_dot,
    I64 *out_dot_q32,
    I64 out_dot_capacity)
{
    I64 pair_idx;
    I64 lhs_pair_base;
    I64 rhs_pair_base;
    I64 lhs_last_offset;
    I64 rhs_last_offset;
    I64 lhs_last_pair_base;
    I64 rhs_last_pair_base;
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 staged_dot_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_dot_capacity < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (pair_count < 0 || blocks_per_dot < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_pair_stride_blocks <= 0 || rhs_pair_stride_blocks <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_block_stride <= 0 || rhs_block_stride <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (pair_count > out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)lhs_q4 == out_dot_q32 || (I64 *)rhs_q8 == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (!pair_count)
        return Q4_0_Q8_0_OK;

    if (!blocks_per_dot)
    {
        for (pair_idx = 0; pair_idx < pair_count; ++pair_idx)
            out_dot_q32[pair_idx] = 0;
        return Q4_0_Q8_0_OK;
    }

    // Checked capacity preflight for the maximum block index touched by the
    // strided array geometry.
    if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1, lhs_block_stride, &lhs_last_offset))
        return Q4_0_Q8_0_ERR_OVERFLOW;
    if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1, rhs_block_stride, &rhs_last_offset))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1, lhs_pair_stride_blocks, &lhs_last_pair_base))
        return Q4_0_Q8_0_ERR_OVERFLOW;
    if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1, rhs_pair_stride_blocks, &rhs_last_pair_base))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (!Q4_0Q8_0TryAddI64(lhs_last_pair_base, lhs_last_offset, &lhs_required_blocks))
        return Q4_0_Q8_0_ERR_OVERFLOW;
    if (!Q4_0Q8_0TryAddI64(rhs_last_pair_base, rhs_last_offset, &rhs_required_blocks))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (!Q4_0Q8_0TryAddI64(lhs_required_blocks, 1, &lhs_required_blocks))
        return Q4_0_Q8_0_ERR_OVERFLOW;
    if (!Q4_0Q8_0TryAddI64(rhs_required_blocks, 1, &rhs_required_blocks))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (lhs_required_blocks > lhs_block_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (rhs_required_blocks > rhs_block_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    // Preflight pass: compute every pair and discard staged results.
    // If any pair faults, out_dot_q32 remains untouched.
    for (pair_idx = 0; pair_idx < pair_count; ++pair_idx)
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_idx, lhs_pair_stride_blocks, &lhs_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_idx, rhs_pair_stride_blocks, &rhs_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        status = Q4_0Q8_0DotQ32CheckedNoPartialArrayDotOne(
            lhs_q4,
            lhs_pair_base,
            lhs_block_stride,
            rhs_q8,
            rhs_pair_base,
            rhs_block_stride,
            blocks_per_dot,
            &staged_dot_q32);
        if (status != Q4_0_Q8_0_OK)
            return status;
    }

    // Commit pass: repeat computations and publish only after full preflight.
    for (pair_idx = 0; pair_idx < pair_count; ++pair_idx)
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_idx, lhs_pair_stride_blocks, &lhs_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_idx, rhs_pair_stride_blocks, &rhs_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        status = Q4_0Q8_0DotQ32CheckedNoPartialArrayDotOne(
            lhs_q4,
            lhs_pair_base,
            lhs_block_stride,
            rhs_q8,
            rhs_pair_base,
            rhs_block_stride,
            blocks_per_dot,
            &staged_dot_q32);
        if (status != Q4_0_Q8_0_OK)
            return status;

        out_dot_q32[pair_idx] = staged_dot_q32;
    }

    return Q4_0_Q8_0_OK;
}



// Commit-only diagnostics wrapper over Q4_0Q8_0DotQ32CheckedNoPartialArray.
//
// Runs checked mixed-dot computation first, then stages canonical geometry
// diagnostics derived from immutable snapshots:
//   pair_count, required_lhs_blocks, required_rhs_blocks,
//   required_out_cells, required_lhs_bytes, required_rhs_bytes, required_out_bytes.
//
// Diagnostics are atomically published only after all checks succeed.
I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly(
    Q4_0Block *lhs_q4,
    I64 lhs_block_capacity,
    I64 lhs_pair_stride_blocks,
    I64 lhs_block_stride,
    Q8_0Block *rhs_q8,
    I64 rhs_block_capacity,
    I64 rhs_pair_stride_blocks,
    I64 rhs_block_stride,
    I64 pair_count,
    I64 blocks_per_dot,
    I64 *out_dot_q32,
    I64 out_dot_capacity,
    I64 *out_pair_count,
    I64 *out_required_lhs_blocks,
    I64 *out_required_rhs_blocks,
    I64 *out_required_out_cells,
    I64 *out_required_lhs_bytes,
    I64 *out_required_rhs_bytes,
    I64 *out_required_out_bytes)
{
    I64 staged_pair_count;
    I64 staged_required_lhs_blocks;
    I64 staged_required_rhs_blocks;
    I64 staged_required_out_cells;
    I64 staged_required_lhs_bytes;
    I64 staged_required_rhs_bytes;
    I64 staged_required_out_bytes;
    I64 lhs_last_offset;
    I64 rhs_last_offset;
    I64 lhs_last_pair_base;
    I64 rhs_last_pair_base;
    I64 snapshot_lhs_block_capacity;
    I64 snapshot_lhs_pair_stride_blocks;
    I64 snapshot_lhs_block_stride;
    I64 snapshot_rhs_block_capacity;
    I64 snapshot_rhs_pair_stride_blocks;
    I64 snapshot_rhs_block_stride;
    I64 snapshot_pair_count;
    I64 snapshot_blocks_per_dot;
    I64 snapshot_out_dot_capacity;
    I64 scratch_out_capacity;
    I64 scratch_out_bytes;
    Q4_0Block *snapshot_lhs_ptr;
    Q8_0Block *snapshot_rhs_ptr;
    I64 *snapshot_out_dot_ptr;
    I64 *scratch_out_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (!out_pair_count ||
        !out_required_lhs_blocks ||
        !out_required_rhs_blocks ||
        !out_required_out_cells ||
        !out_required_lhs_bytes ||
        !out_required_rhs_bytes ||
        !out_required_out_bytes)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (out_pair_count == out_required_lhs_blocks ||
        out_pair_count == out_required_rhs_blocks ||
        out_pair_count == out_required_out_cells ||
        out_pair_count == out_required_lhs_bytes ||
        out_pair_count == out_required_rhs_bytes ||
        out_pair_count == out_required_out_bytes ||
        out_required_lhs_blocks == out_required_rhs_blocks ||
        out_required_lhs_blocks == out_required_out_cells ||
        out_required_lhs_blocks == out_required_lhs_bytes ||
        out_required_lhs_blocks == out_required_rhs_bytes ||
        out_required_lhs_blocks == out_required_out_bytes ||
        out_required_rhs_blocks == out_required_out_cells ||
        out_required_rhs_blocks == out_required_lhs_bytes ||
        out_required_rhs_blocks == out_required_rhs_bytes ||
        out_required_rhs_blocks == out_required_out_bytes ||
        out_required_out_cells == out_required_lhs_bytes ||
        out_required_out_cells == out_required_rhs_bytes ||
        out_required_out_cells == out_required_out_bytes ||
        out_required_lhs_bytes == out_required_rhs_bytes ||
        out_required_lhs_bytes == out_required_out_bytes ||
        out_required_rhs_bytes == out_required_out_bytes)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (out_pair_count == (I64 *)lhs_q4 ||
        out_pair_count == (I64 *)rhs_q8 ||
        out_pair_count == out_dot_q32 ||
        out_required_lhs_blocks == (I64 *)lhs_q4 ||
        out_required_lhs_blocks == (I64 *)rhs_q8 ||
        out_required_lhs_blocks == out_dot_q32 ||
        out_required_rhs_blocks == (I64 *)lhs_q4 ||
        out_required_rhs_blocks == (I64 *)rhs_q8 ||
        out_required_rhs_blocks == out_dot_q32 ||
        out_required_out_cells == (I64 *)lhs_q4 ||
        out_required_out_cells == (I64 *)rhs_q8 ||
        out_required_out_cells == out_dot_q32 ||
        out_required_lhs_bytes == (I64 *)lhs_q4 ||
        out_required_lhs_bytes == (I64 *)rhs_q8 ||
        out_required_lhs_bytes == out_dot_q32 ||
        out_required_rhs_bytes == (I64 *)lhs_q4 ||
        out_required_rhs_bytes == (I64 *)rhs_q8 ||
        out_required_rhs_bytes == out_dot_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4 ||
        out_required_out_bytes == (I64 *)rhs_q8 ||
        out_required_out_bytes == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs_q4;
    snapshot_rhs_ptr = rhs_q8;
    snapshot_out_dot_ptr = out_dot_q32;
    snapshot_lhs_block_capacity = lhs_block_capacity;
    snapshot_lhs_pair_stride_blocks = lhs_pair_stride_blocks;
    snapshot_lhs_block_stride = lhs_block_stride;
    snapshot_rhs_block_capacity = rhs_block_capacity;
    snapshot_rhs_pair_stride_blocks = rhs_pair_stride_blocks;
    snapshot_rhs_block_stride = rhs_block_stride;
    snapshot_pair_count = pair_count;
    snapshot_blocks_per_dot = blocks_per_dot;
    snapshot_out_dot_capacity = out_dot_capacity;

    status = Q4_0Q8_0DotQ32CheckedNoPartialArray(
        lhs_q4,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_q8,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        out_dot_q32,
        out_dot_capacity);
    if (status != Q4_0_Q8_0_OK)
        return status;

    staged_pair_count = pair_count;
    staged_required_out_cells = pair_count;
    if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_out_cells,
                                 sizeof(I64),
                                 &staged_required_out_bytes))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (!pair_count || !blocks_per_dot)
    {
        staged_required_lhs_blocks = 0;
        staged_required_rhs_blocks = 0;
        staged_required_lhs_bytes = 0;
        staged_required_rhs_bytes = 0;
    }
    else
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1,
                                     lhs_block_stride,
                                     &lhs_last_offset))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1,
                                     rhs_block_stride,
                                     &rhs_last_offset))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1,
                                     lhs_pair_stride_blocks,
                                     &lhs_last_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1,
                                     rhs_pair_stride_blocks,
                                     &rhs_last_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryAddI64(lhs_last_pair_base,
                               lhs_last_offset,
                               &staged_required_lhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(rhs_last_pair_base,
                               rhs_last_offset,
                               &staged_required_rhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryAddI64(staged_required_lhs_blocks,
                               1,
                               &staged_required_lhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(staged_required_rhs_blocks,
                               1,
                               &staged_required_rhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (staged_required_lhs_blocks > lhs_block_capacity)
            return Q4_0_Q8_0_ERR_BAD_DST_LEN;
        if (staged_required_rhs_blocks > rhs_block_capacity)
            return Q4_0_Q8_0_ERR_BAD_DST_LEN;

        if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_lhs_blocks,
                                     sizeof(Q4_0Block),
                                     &staged_required_lhs_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_rhs_blocks,
                                     sizeof(Q8_0Block),
                                     &staged_required_rhs_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    if (snapshot_lhs_ptr != lhs_q4 ||
        snapshot_rhs_ptr != rhs_q8 ||
        snapshot_out_dot_ptr != out_dot_q32 ||
        snapshot_lhs_block_capacity != lhs_block_capacity ||
        snapshot_lhs_pair_stride_blocks != lhs_pair_stride_blocks ||
        snapshot_lhs_block_stride != lhs_block_stride ||
        snapshot_rhs_block_capacity != rhs_block_capacity ||
        snapshot_rhs_pair_stride_blocks != rhs_pair_stride_blocks ||
        snapshot_rhs_block_stride != rhs_block_stride ||
        snapshot_pair_count != pair_count ||
        snapshot_blocks_per_dot != blocks_per_dot ||
        snapshot_out_dot_capacity != out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    *out_pair_count = staged_pair_count;
    *out_required_lhs_blocks = staged_required_lhs_blocks;
    *out_required_rhs_blocks = staged_required_rhs_blocks;
    *out_required_out_cells = staged_required_out_cells;
    *out_required_lhs_bytes = staged_required_lhs_bytes;
    *out_required_rhs_bytes = staged_required_rhs_bytes;
    *out_required_out_bytes = staged_required_out_bytes;
    return Q4_0_Q8_0_OK;
}

// Diagnostics-only no-write companion for mixed Q4_0 x Q8_0 checked-array dots.
//
// This helper validates immutable geometry/pointer snapshots and publishes only
// canonical required-size diagnostics:
//   count               = pair_count
//   required_in_blocks  = max(required_lhs_blocks, required_rhs_blocks)
//   required_out_cells  = pair_count
//   required_in_bytes   = required_lhs_blocks*sizeof(Q4_0Block) + required_rhs_blocks*sizeof(Q8_0Block)
//   required_out_bytes  = required_out_cells*sizeof(I64)
//
// It never writes caller dot outputs; diagnostics commit atomically only after
// full checked preflight and snapshot revalidation succeed.
I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnlyPreflightOnly(
    Q4_0Block *lhs_q4,
    I64 lhs_block_capacity,
    I64 lhs_pair_stride_blocks,
    I64 lhs_block_stride,
    Q8_0Block *rhs_q8,
    I64 rhs_block_capacity,
    I64 rhs_pair_stride_blocks,
    I64 rhs_block_stride,
    I64 pair_count,
    I64 blocks_per_dot,
    I64 *out_dot_q32,
    I64 out_dot_capacity,
    I64 *out_count,
    I64 *out_required_in_blocks,
    I64 *out_required_out_cells,
    I64 *out_required_in_bytes,
    I64 *out_required_out_bytes)
{
    I64 staged_count;
    I64 staged_required_in_blocks;
    I64 staged_required_out_cells;
    I64 staged_required_in_bytes;
    I64 staged_required_out_bytes;
    I64 lhs_last_offset;
    I64 rhs_last_offset;
    I64 lhs_last_pair_base;
    I64 rhs_last_pair_base;
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 lhs_required_bytes;
    I64 rhs_required_bytes;
    I64 snapshot_lhs_block_capacity;
    I64 snapshot_lhs_pair_stride_blocks;
    I64 snapshot_lhs_block_stride;
    I64 snapshot_rhs_block_capacity;
    I64 snapshot_rhs_pair_stride_blocks;
    I64 snapshot_rhs_block_stride;
    I64 snapshot_pair_count;
    I64 snapshot_blocks_per_dot;
    I64 snapshot_out_dot_capacity;
    Q4_0Block *snapshot_lhs_ptr;
    Q8_0Block *snapshot_rhs_ptr;
    I64 *snapshot_out_dot_ptr;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (!out_count || !out_required_in_blocks || !out_required_out_cells ||
        !out_required_in_bytes || !out_required_out_bytes)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (out_count == out_required_in_blocks ||
        out_count == out_required_out_cells ||
        out_count == out_required_in_bytes ||
        out_count == out_required_out_bytes ||
        out_required_in_blocks == out_required_out_cells ||
        out_required_in_blocks == out_required_in_bytes ||
        out_required_in_blocks == out_required_out_bytes ||
        out_required_out_cells == out_required_in_bytes ||
        out_required_out_cells == out_required_out_bytes ||
        out_required_in_bytes == out_required_out_bytes)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (out_count == (I64 *)lhs_q4 ||
        out_count == (I64 *)rhs_q8 ||
        out_count == out_dot_q32 ||
        out_required_in_blocks == (I64 *)lhs_q4 ||
        out_required_in_blocks == (I64 *)rhs_q8 ||
        out_required_in_blocks == out_dot_q32 ||
        out_required_out_cells == (I64 *)lhs_q4 ||
        out_required_out_cells == (I64 *)rhs_q8 ||
        out_required_out_cells == out_dot_q32 ||
        out_required_in_bytes == (I64 *)lhs_q4 ||
        out_required_in_bytes == (I64 *)rhs_q8 ||
        out_required_in_bytes == out_dot_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4 ||
        out_required_out_bytes == (I64 *)rhs_q8 ||
        out_required_out_bytes == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (out_count == (I64 *)lhs_q4 ||
        out_count == (I64 *)rhs_q8 ||
        out_count == out_dot_q32 ||
        out_required_in_blocks == (I64 *)lhs_q4 ||
        out_required_in_blocks == (I64 *)rhs_q8 ||
        out_required_in_blocks == out_dot_q32 ||
        out_required_out_cells == (I64 *)lhs_q4 ||
        out_required_out_cells == (I64 *)rhs_q8 ||
        out_required_out_cells == out_dot_q32 ||
        out_required_in_bytes == (I64 *)lhs_q4 ||
        out_required_in_bytes == (I64 *)rhs_q8 ||
        out_required_in_bytes == out_dot_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4 ||
        out_required_out_bytes == (I64 *)rhs_q8 ||
        out_required_out_bytes == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (out_count == (I64 *)lhs_q4 ||
        out_count == (I64 *)rhs_q8 ||
        out_count == out_dot_q32 ||
        out_required_in_blocks == (I64 *)lhs_q4 ||
        out_required_in_blocks == (I64 *)rhs_q8 ||
        out_required_in_blocks == out_dot_q32 ||
        out_required_out_cells == (I64 *)lhs_q4 ||
        out_required_out_cells == (I64 *)rhs_q8 ||
        out_required_out_cells == out_dot_q32 ||
        out_required_in_bytes == (I64 *)lhs_q4 ||
        out_required_in_bytes == (I64 *)rhs_q8 ||
        out_required_in_bytes == out_dot_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4 ||
        out_required_out_bytes == (I64 *)rhs_q8 ||
        out_required_out_bytes == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_dot_capacity < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (pair_count < 0 || blocks_per_dot < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_pair_stride_blocks <= 0 || rhs_pair_stride_blocks <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_block_stride <= 0 || rhs_block_stride <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (pair_count > out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs_q4;
    snapshot_rhs_ptr = rhs_q8;
    snapshot_out_dot_ptr = out_dot_q32;
    snapshot_lhs_block_capacity = lhs_block_capacity;
    snapshot_lhs_pair_stride_blocks = lhs_pair_stride_blocks;
    snapshot_lhs_block_stride = lhs_block_stride;
    snapshot_rhs_block_capacity = rhs_block_capacity;
    snapshot_rhs_pair_stride_blocks = rhs_pair_stride_blocks;
    snapshot_rhs_block_stride = rhs_block_stride;
    snapshot_pair_count = pair_count;
    snapshot_blocks_per_dot = blocks_per_dot;
    snapshot_out_dot_capacity = out_dot_capacity;
    scratch_out_q32 = NULL;
    scratch_out_capacity = pair_count;
    scratch_out_bytes = 0;

    staged_count = pair_count;
    staged_required_out_cells = pair_count;
    if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_out_cells,
                                 sizeof(I64),
                                 &staged_required_out_bytes))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (!pair_count || !blocks_per_dot)
    {
        staged_required_in_blocks = 0;
        staged_required_in_bytes = 0;
    }
    else
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1,
                                     lhs_block_stride,
                                     &lhs_last_offset))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1,
                                     rhs_block_stride,
                                     &rhs_last_offset))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1,
                                     lhs_pair_stride_blocks,
                                     &lhs_last_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1,
                                     rhs_pair_stride_blocks,
                                     &rhs_last_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryAddI64(lhs_last_pair_base,
                               lhs_last_offset,
                               &lhs_required_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(rhs_last_pair_base,
                               rhs_last_offset,
                               &rhs_required_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryAddI64(lhs_required_blocks, 1, &lhs_required_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(rhs_required_blocks, 1, &rhs_required_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (lhs_required_blocks > lhs_block_capacity)
            return Q4_0_Q8_0_ERR_BAD_DST_LEN;
        if (rhs_required_blocks > rhs_block_capacity)
            return Q4_0_Q8_0_ERR_BAD_DST_LEN;

        if (lhs_required_blocks >= rhs_required_blocks)
            staged_required_in_blocks = lhs_required_blocks;
        else
            staged_required_in_blocks = rhs_required_blocks;

        if (!Q4_0Q8_0TryMulI64NonNeg(lhs_required_blocks,
                                     sizeof(Q4_0Block),
                                     &lhs_required_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(rhs_required_blocks,
                                     sizeof(Q8_0Block),
                                     &rhs_required_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(lhs_required_bytes,
                               rhs_required_bytes,
                               &staged_required_in_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    if (snapshot_lhs_ptr != lhs_q4 ||
        snapshot_rhs_ptr != rhs_q8 ||
        snapshot_out_dot_ptr != out_dot_q32 ||
        snapshot_lhs_block_capacity != lhs_block_capacity ||
        snapshot_lhs_pair_stride_blocks != lhs_pair_stride_blocks ||
        snapshot_lhs_block_stride != lhs_block_stride ||
        snapshot_rhs_block_capacity != rhs_block_capacity ||
        snapshot_rhs_pair_stride_blocks != rhs_pair_stride_blocks ||
        snapshot_rhs_block_stride != rhs_block_stride ||
        snapshot_pair_count != pair_count ||
        snapshot_blocks_per_dot != blocks_per_dot ||
        snapshot_out_dot_capacity != out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    *out_count = staged_count;
    *out_required_in_blocks = staged_required_in_blocks;
    *out_required_out_cells = staged_required_out_cells;
    *out_required_in_bytes = staged_required_in_bytes;
    *out_required_out_bytes = staged_required_out_bytes;
    return Q4_0_Q8_0_OK;
}

I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnlyPreflightOnlyParity(
    Q4_0Block *lhs_q4,
    I64 lhs_block_capacity,
    I64 lhs_pair_stride_blocks,
    I64 lhs_block_stride,
    Q8_0Block *rhs_q8,
    I64 rhs_block_capacity,
    I64 rhs_pair_stride_blocks,
    I64 rhs_block_stride,
    I64 pair_count,
    I64 blocks_per_dot,
    I64 *out_dot_q32,
    I64 out_dot_capacity,
    I64 *out_count,
    I64 *out_required_in_blocks,
    I64 *out_required_out_cells,
    I64 *out_required_in_bytes,
    I64 *out_required_out_bytes)
{
    I64 staged_count;
    I64 staged_required_in_blocks;
    I64 staged_required_out_cells;
    I64 staged_required_in_bytes;
    I64 staged_required_out_bytes;
    I64 recomputed_count;
    I64 recomputed_required_in_blocks;
    I64 recomputed_required_out_cells;
    I64 recomputed_required_in_bytes;
    I64 recomputed_required_out_bytes;
    I64 snapshot_lhs_block_capacity;
    I64 snapshot_lhs_pair_stride_blocks;
    I64 snapshot_lhs_block_stride;
    I64 snapshot_rhs_block_capacity;
    I64 snapshot_rhs_pair_stride_blocks;
    I64 snapshot_rhs_block_stride;
    I64 snapshot_pair_count;
    I64 snapshot_blocks_per_dot;
    I64 snapshot_out_dot_capacity;
    I64 scratch_out_capacity;
    I64 scratch_out_bytes;
    Q4_0Block *snapshot_lhs_ptr;
    Q8_0Block *snapshot_rhs_ptr;
    I64 *snapshot_out_dot_ptr;
    I64 *scratch_out_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (!out_count || !out_required_in_blocks || !out_required_out_cells ||
        !out_required_in_bytes || !out_required_out_bytes)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (out_count == out_required_in_blocks ||
        out_count == out_required_out_cells ||
        out_count == out_required_in_bytes ||
        out_count == out_required_out_bytes ||
        out_required_in_blocks == out_required_out_cells ||
        out_required_in_blocks == out_required_in_bytes ||
        out_required_in_blocks == out_required_out_bytes ||
        out_required_out_cells == out_required_in_bytes ||
        out_required_out_cells == out_required_out_bytes ||
        out_required_in_bytes == out_required_out_bytes)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (out_count == (I64 *)lhs_q4 ||
        out_count == (I64 *)rhs_q8 ||
        out_count == out_dot_q32 ||
        out_required_in_blocks == (I64 *)lhs_q4 ||
        out_required_in_blocks == (I64 *)rhs_q8 ||
        out_required_in_blocks == out_dot_q32 ||
        out_required_out_cells == (I64 *)lhs_q4 ||
        out_required_out_cells == (I64 *)rhs_q8 ||
        out_required_out_cells == out_dot_q32 ||
        out_required_in_bytes == (I64 *)lhs_q4 ||
        out_required_in_bytes == (I64 *)rhs_q8 ||
        out_required_in_bytes == out_dot_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4 ||
        out_required_out_bytes == (I64 *)rhs_q8 ||
        out_required_out_bytes == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_dot_capacity < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (pair_count < 0 || blocks_per_dot < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_pair_stride_blocks <= 0 || rhs_pair_stride_blocks <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_block_stride <= 0 || rhs_block_stride <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (pair_count > out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs_q4;
    snapshot_rhs_ptr = rhs_q8;
    snapshot_out_dot_ptr = out_dot_q32;
    snapshot_lhs_block_capacity = lhs_block_capacity;
    snapshot_lhs_pair_stride_blocks = lhs_pair_stride_blocks;
    snapshot_lhs_block_stride = lhs_block_stride;
    snapshot_rhs_block_capacity = rhs_block_capacity;
    snapshot_rhs_pair_stride_blocks = rhs_pair_stride_blocks;
    snapshot_rhs_block_stride = rhs_block_stride;
    snapshot_pair_count = pair_count;
    snapshot_blocks_per_dot = blocks_per_dot;
    snapshot_out_dot_capacity = out_dot_capacity;
    scratch_out_q32 = NULL;
    scratch_out_capacity = pair_count;
    scratch_out_bytes = 0;

    // Stage diagnostics from the no-write preflight companion.
    status = Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnlyPreflightOnly(
        lhs_q4,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_q8,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        out_dot_q32,
        out_dot_capacity,
        &staged_count,
        &staged_required_in_blocks,
        &staged_required_out_cells,
        &staged_required_in_bytes,
        &staged_required_out_bytes);
    if (status != Q4_0_Q8_0_OK)
        return status;

    // Compose against canonical checked-array execution but keep caller
    // out_dot_q32 untouched by executing into a private scratch buffer.
    if (pair_count > 0)
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count,
                                     sizeof(I64),
                                     &scratch_out_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        scratch_out_q32 = MAlloc(scratch_out_bytes);
        if (!scratch_out_q32)
            return Q4_0_Q8_0_ERR_NULL_PTR;
    }

    // Compose against canonical checked-array execution and recompute expected
    // diagnostics from immutable geometry for exact tuple parity.
    status = Q4_0Q8_0DotQ32CheckedNoPartialArray(
        lhs_q4,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_q8,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        pair_count > 0 ? scratch_out_q32 : out_dot_q32,
        pair_count > 0 ? scratch_out_capacity : out_dot_capacity);
    if (status != Q4_0_Q8_0_OK)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return status;
    }

    status = Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnlyPreflightOnly(
        lhs_q4,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_q8,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        out_dot_q32,
        out_dot_capacity,
        &recomputed_count,
        &recomputed_required_in_blocks,
        &recomputed_required_out_cells,
        &recomputed_required_in_bytes,
        &recomputed_required_out_bytes);
    if (status != Q4_0_Q8_0_OK)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return status;
    }

    if (snapshot_lhs_ptr != lhs_q4 ||
        snapshot_rhs_ptr != rhs_q8 ||
        snapshot_out_dot_ptr != out_dot_q32 ||
        snapshot_lhs_block_capacity != lhs_block_capacity ||
        snapshot_lhs_pair_stride_blocks != lhs_pair_stride_blocks ||
        snapshot_lhs_block_stride != lhs_block_stride ||
        snapshot_rhs_block_capacity != rhs_block_capacity ||
        snapshot_rhs_pair_stride_blocks != rhs_pair_stride_blocks ||
        snapshot_rhs_block_stride != rhs_block_stride ||
        snapshot_pair_count != pair_count ||
        snapshot_blocks_per_dot != blocks_per_dot ||
        snapshot_out_dot_capacity != out_dot_capacity)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    }

    if (staged_count != recomputed_count)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    }
    if (staged_required_in_blocks != recomputed_required_in_blocks)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    }
    if (staged_required_out_cells != recomputed_required_out_cells)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    }
    if (staged_required_in_bytes != recomputed_required_in_bytes)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    }
    if (staged_required_out_bytes != recomputed_required_out_bytes)
    {
        if (scratch_out_q32)
            Free(scratch_out_q32);
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    }

    *out_count = staged_count;
    *out_required_in_blocks = staged_required_in_blocks;
    *out_required_out_cells = staged_required_out_cells;
    *out_required_in_bytes = staged_required_in_bytes;
    *out_required_out_bytes = staged_required_out_bytes;

    if (scratch_out_q32)
        Free(scratch_out_q32);
    return Q4_0_Q8_0_OK;
}

// Commit-only diagnostics wrapper for checked mixed-array dots.
//
// This helper executes Q4_0Q8_0DotQ32CheckedNoPartialArray to publish dot
// outputs, then stages canonical required-geometry diagnostics and commits
// those diagnostics atomically after immutable snapshot revalidation.
I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly(
    Q4_0Block *lhs_q4,
    I64 lhs_block_capacity,
    I64 lhs_pair_stride_blocks,
    I64 lhs_block_stride,
    Q8_0Block *rhs_q8,
    I64 rhs_block_capacity,
    I64 rhs_pair_stride_blocks,
    I64 rhs_block_stride,
    I64 pair_count,
    I64 blocks_per_dot,
    I64 *out_dot_q32,
    I64 out_dot_capacity,
    I64 *out_pair_count,
    I64 *out_required_lhs_blocks,
    I64 *out_required_rhs_blocks,
    I64 *out_required_out_cells,
    I64 *out_required_lhs_bytes,
    I64 *out_required_rhs_bytes,
    I64 *out_required_out_bytes)
{
    I64 staged_pair_count;
    I64 staged_required_lhs_blocks;
    I64 staged_required_rhs_blocks;
    I64 staged_required_out_cells;
    I64 staged_required_lhs_bytes;
    I64 staged_required_rhs_bytes;
    I64 staged_required_out_bytes;
    I64 lhs_last_offset;
    I64 rhs_last_offset;
    I64 lhs_last_pair_base;
    I64 rhs_last_pair_base;
    I64 snapshot_lhs_block_capacity;
    I64 snapshot_lhs_pair_stride_blocks;
    I64 snapshot_lhs_block_stride;
    I64 snapshot_rhs_block_capacity;
    I64 snapshot_rhs_pair_stride_blocks;
    I64 snapshot_rhs_block_stride;
    I64 snapshot_pair_count;
    I64 snapshot_blocks_per_dot;
    I64 snapshot_out_dot_capacity;
    I64 scratch_out_capacity;
    I64 scratch_out_bytes;
    Q4_0Block *snapshot_lhs_ptr;
    Q8_0Block *snapshot_rhs_ptr;
    I64 *snapshot_out_dot_ptr;
    I64 *scratch_out_q32;
    I32 status;

    if (!lhs_q4 || !rhs_q8 || !out_dot_q32)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (!out_pair_count || !out_required_lhs_blocks || !out_required_rhs_blocks ||
        !out_required_out_cells || !out_required_lhs_bytes || !out_required_rhs_bytes ||
        !out_required_out_bytes)
        return Q4_0_Q8_0_ERR_NULL_PTR;

    if (out_pair_count == out_required_lhs_blocks ||
        out_pair_count == out_required_rhs_blocks ||
        out_pair_count == out_required_out_cells ||
        out_pair_count == out_required_lhs_bytes ||
        out_pair_count == out_required_rhs_bytes ||
        out_pair_count == out_required_out_bytes ||
        out_required_lhs_blocks == out_required_rhs_blocks ||
        out_required_lhs_blocks == out_required_out_cells ||
        out_required_lhs_blocks == out_required_lhs_bytes ||
        out_required_lhs_blocks == out_required_rhs_bytes ||
        out_required_lhs_blocks == out_required_out_bytes ||
        out_required_rhs_blocks == out_required_out_cells ||
        out_required_rhs_blocks == out_required_lhs_bytes ||
        out_required_rhs_blocks == out_required_rhs_bytes ||
        out_required_rhs_blocks == out_required_out_bytes ||
        out_required_out_cells == out_required_lhs_bytes ||
        out_required_out_cells == out_required_rhs_bytes ||
        out_required_out_cells == out_required_out_bytes ||
        out_required_lhs_bytes == out_required_rhs_bytes ||
        out_required_lhs_bytes == out_required_out_bytes ||
        out_required_rhs_bytes == out_required_out_bytes)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (out_pair_count == (I64 *)lhs_q4 ||
        out_pair_count == (I64 *)rhs_q8 ||
        out_pair_count == out_dot_q32 ||
        out_required_lhs_blocks == (I64 *)lhs_q4 ||
        out_required_lhs_blocks == (I64 *)rhs_q8 ||
        out_required_lhs_blocks == out_dot_q32 ||
        out_required_rhs_blocks == (I64 *)lhs_q4 ||
        out_required_rhs_blocks == (I64 *)rhs_q8 ||
        out_required_rhs_blocks == out_dot_q32 ||
        out_required_out_cells == (I64 *)lhs_q4 ||
        out_required_out_cells == (I64 *)rhs_q8 ||
        out_required_out_cells == out_dot_q32 ||
        out_required_lhs_bytes == (I64 *)lhs_q4 ||
        out_required_lhs_bytes == (I64 *)rhs_q8 ||
        out_required_lhs_bytes == out_dot_q32 ||
        out_required_rhs_bytes == (I64 *)lhs_q4 ||
        out_required_rhs_bytes == (I64 *)rhs_q8 ||
        out_required_rhs_bytes == out_dot_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4 ||
        out_required_out_bytes == (I64 *)rhs_q8 ||
        out_required_out_bytes == out_dot_q32)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_dot_capacity < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (pair_count < 0 || blocks_per_dot < 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_pair_stride_blocks <= 0 || rhs_pair_stride_blocks <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (lhs_block_stride <= 0 || rhs_block_stride <= 0)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;
    if (pair_count > out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs_q4;
    snapshot_rhs_ptr = rhs_q8;
    snapshot_out_dot_ptr = out_dot_q32;
    snapshot_lhs_block_capacity = lhs_block_capacity;
    snapshot_lhs_pair_stride_blocks = lhs_pair_stride_blocks;
    snapshot_lhs_block_stride = lhs_block_stride;
    snapshot_rhs_block_capacity = rhs_block_capacity;
    snapshot_rhs_pair_stride_blocks = rhs_pair_stride_blocks;
    snapshot_rhs_block_stride = rhs_block_stride;
    snapshot_pair_count = pair_count;
    snapshot_blocks_per_dot = blocks_per_dot;
    snapshot_out_dot_capacity = out_dot_capacity;

    status = Q4_0Q8_0DotQ32CheckedNoPartialArray(
        lhs_q4,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_q8,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        pair_count > 0 ? scratch_out_q32 : out_dot_q32,
        pair_count > 0 ? scratch_out_capacity : out_dot_capacity);
    if (status != Q4_0_Q8_0_OK)
        return status;

    staged_pair_count = pair_count;
    staged_required_out_cells = pair_count;
    if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_out_cells,
                                 sizeof(I64),
                                 &staged_required_out_bytes))
        return Q4_0_Q8_0_ERR_OVERFLOW;

    if (!pair_count || !blocks_per_dot)
    {
        staged_required_lhs_blocks = 0;
        staged_required_rhs_blocks = 0;
        staged_required_lhs_bytes = 0;
        staged_required_rhs_bytes = 0;
    }
    else
    {
        if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1,
                                     lhs_block_stride,
                                     &lhs_last_offset))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(blocks_per_dot - 1,
                                     rhs_block_stride,
                                     &rhs_last_offset))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1,
                                     lhs_pair_stride_blocks,
                                     &lhs_last_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(pair_count - 1,
                                     rhs_pair_stride_blocks,
                                     &rhs_last_pair_base))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryAddI64(lhs_last_pair_base,
                               lhs_last_offset,
                               &staged_required_lhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(rhs_last_pair_base,
                               rhs_last_offset,
                               &staged_required_rhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (!Q4_0Q8_0TryAddI64(staged_required_lhs_blocks, 1, &staged_required_lhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryAddI64(staged_required_rhs_blocks, 1, &staged_required_rhs_blocks))
            return Q4_0_Q8_0_ERR_OVERFLOW;

        if (staged_required_lhs_blocks > lhs_block_capacity)
            return Q4_0_Q8_0_ERR_BAD_DST_LEN;
        if (staged_required_rhs_blocks > rhs_block_capacity)
            return Q4_0_Q8_0_ERR_BAD_DST_LEN;

        if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_lhs_blocks,
                                     sizeof(Q4_0Block),
                                     &staged_required_lhs_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
        if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_rhs_blocks,
                                     sizeof(Q8_0Block),
                                     &staged_required_rhs_bytes))
            return Q4_0_Q8_0_ERR_OVERFLOW;
    }

    if (snapshot_lhs_ptr != lhs_q4 ||
        snapshot_rhs_ptr != rhs_q8 ||
        snapshot_out_dot_ptr != out_dot_q32 ||
        snapshot_lhs_block_capacity != lhs_block_capacity ||
        snapshot_lhs_pair_stride_blocks != lhs_pair_stride_blocks ||
        snapshot_lhs_block_stride != lhs_block_stride ||
        snapshot_rhs_block_capacity != rhs_block_capacity ||
        snapshot_rhs_pair_stride_blocks != rhs_pair_stride_blocks ||
        snapshot_rhs_block_stride != rhs_block_stride ||
        snapshot_pair_count != pair_count ||
        snapshot_blocks_per_dot != blocks_per_dot ||
        snapshot_out_dot_capacity != out_dot_capacity)
        return Q4_0_Q8_0_ERR_BAD_DST_LEN;

    *out_pair_count = staged_pair_count;
    *out_required_lhs_blocks = staged_required_lhs_blocks;
    *out_required_rhs_blocks = staged_required_rhs_blocks;
    *out_required_out_cells = staged_required_out_cells;
    *out_required_lhs_bytes = staged_required_lhs_bytes;
    *out_required_rhs_bytes = staged_required_rhs_bytes;
    *out_required_out_bytes = staged_required_out_bytes;
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
